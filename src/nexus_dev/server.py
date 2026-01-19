"""Nexus-Dev MCP Server.

This module implements the MCP server using FastMCP, exposing tools for:
- search_code: Semantic search across indexed code and documentation
- index_file: Index a file into the knowledge base
- record_lesson: Store a problem/solution pair
- get_project_context: Get recent discoveries for a project
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context

from .agents import AgentConfig, AgentExecutor, AgentManager
from .chunkers import ChunkerRegistry, ChunkType, CodeChunk
from .config import NexusConfig
from .database import Document, DocumentType, NexusDatabase, generate_document_id
from .embeddings import EmbeddingProvider, create_embedder
from .gateway.connection_manager import ConnectionManager
from .github_importer import GitHubImporter
from .mcp_config import MCPConfig

# Initialize FastMCP server
mcp = FastMCP("nexus-dev")

logger = logging.getLogger(__name__)

# Global state (initialized on startup)
_config: NexusConfig | None = None
_embedder: EmbeddingProvider | None = None
_database: NexusDatabase | None = None
_mcp_config: MCPConfig | None = None
_connection_manager: ConnectionManager | None = None
_agent_manager: AgentManager | None = None
_project_root: Path | None = None


def _find_project_root() -> Path | None:
    """Find the project root by looking for nexus_config.json.

    Walks up from the current directory to find nexus_config.json.
    Also checks NEXUS_PROJECT_ROOT environment variable as a fallback.

    Returns:
        Path to project root if found, None otherwise.
    """
    global _project_root
    if _project_root:
        return _project_root

    import os

    # First check environment variable
    env_root = os.environ.get("NEXUS_PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root)
        if (env_path / "nexus_config.json").exists():
            logger.debug("Found project root from NEXUS_PROJECT_ROOT: %s", env_path)
            return env_path

    current = Path.cwd().resolve()
    logger.debug("Searching for project root from cwd: %s", current)

    # Walk up the directory tree
    for parent in [current] + list(current.parents):
        if (parent / "nexus_config.json").exists():
            logger.debug("Found project root: %s", parent)
            _project_root = parent
            return parent
        # Stop at filesystem root
        if parent == parent.parent:
            logger.debug("Reached filesystem root without finding nexus_config.json")
            break

    logger.debug("No project root found (no nexus_config.json in directory tree)")
    return None


def _get_config() -> NexusConfig | None:
    """Get or load configuration.

    Returns None if no nexus_config.json exists in cwd.
    This allows the MCP server to work without a project-specific config,
    enabling cross-project searches.
    """
    global _config
    if _config is None:
        root = _find_project_root()
        config_path = (root if root else Path.cwd()) / "nexus_config.json"
        if config_path.exists():
            _config = NexusConfig.load(config_path)
        # Don't create default - None means "all projects"
    return _config


def _get_mcp_config() -> MCPConfig | None:
    """Get or load MCP configuration.

    Returns None if no .nexus/mcp_config.json exists in cwd or project root.
    """
    global _mcp_config
    if _mcp_config is None:
        root = _find_project_root()
        config_path = (root if root else Path.cwd()) / ".nexus" / "mcp_config.json"
        if config_path.exists():
            try:
                _mcp_config = MCPConfig.load(config_path)
            except Exception as e:
                logger.debug("Failed to load MCP config from %s: %s", config_path, e)
                pass
    return _mcp_config


def _get_active_server_names() -> list[str]:
    """Get names of active MCP servers.

    Returns:
        List of active server names.
    """
    mcp_config = _get_mcp_config()
    if not mcp_config:
        return []

    # Find the name for each active server config
    active_servers = mcp_config.get_active_servers()
    active_names = []
    for name, config in mcp_config.servers.items():
        if config in active_servers:
            active_names.append(name)
    return active_names


def _get_connection_manager() -> ConnectionManager:
    """Get or create connection manager singleton.

    Returns:
        ConnectionManager instance for managing MCP server connections.
    """
    global _connection_manager
    if _connection_manager is None:
        mcp_config = _get_mcp_config()
        if mcp_config is not None:
            _connection_manager = ConnectionManager(
                default_max_concurrent=mcp_config.gateway.max_concurrent_connections,
                shutdown_timeout=mcp_config.gateway.shutdown_timeout,
            )
        else:
            _connection_manager = ConnectionManager()
    return _connection_manager


def _get_embedder() -> EmbeddingProvider:
    """Get or create embedding provider."""
    global _embedder
    if _embedder is None:
        config = _get_config()
        if config is None:
            # Create minimal config for embeddings only
            config = NexusConfig.create_new("default")
        _embedder = create_embedder(config)
    return _embedder


def _get_database() -> NexusDatabase:
    """Get or create database connection."""
    global _database
    if _database is None:
        config = _get_config()
        if config is None:
            # Create minimal config for database access
            config = NexusConfig.create_new("default")
        embedder = _get_embedder()
        _database = NexusDatabase(config, embedder)
        _database.connect()
    return _database


async def _index_chunks(
    chunks: list[CodeChunk],
    project_id: str,
    doc_type: DocumentType,
) -> list[str]:
    """Index a list of chunks into the database.

    Args:
        chunks: Code chunks to index.
        project_id: Project identifier.
        doc_type: Type of document.

    Returns:
        List of document IDs.
    """
    if not chunks:
        return []

    embedder = _get_embedder()
    database = _get_database()

    # Generate embeddings for all chunks
    texts = [chunk.get_searchable_text() for chunk in chunks]
    embeddings = await embedder.embed_batch(texts)

    # Create documents
    documents = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        doc_id = generate_document_id(
            project_id,
            chunk.file_path,
            chunk.name,
            chunk.start_line,
        )

        doc = Document(
            id=doc_id,
            text=chunk.get_searchable_text(),
            vector=embedding,
            project_id=project_id,
            file_path=chunk.file_path,
            doc_type=doc_type,
            chunk_type=chunk.chunk_type.value,
            language=chunk.language,
            name=chunk.name,
            start_line=chunk.start_line,
            end_line=chunk.end_line,
        )
        documents.append(doc)

    # Upsert documents
    return await database.upsert_documents(documents)


@mcp.tool()
async def search_knowledge(
    query: str,
    content_type: str = "all",
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search all indexed knowledge including code, documentation, and lessons.

    This is the main search tool that can find relevant information across all
    indexed content types. Use the content_type parameter to filter results.

    Args:
        query: Natural language search query describing what you're looking for.
               Examples: "function that handles user authentication",
               "how to configure the database", "error with null pointer".
        content_type: Filter by content type. Options:
                     - "all": Search everything (default)
                     - "code": Only search code (functions, classes, methods)
                     - "documentation": Only search docs (markdown, rst, txt)
                     - "lesson": Only search recorded lessons
        project_id: Optional project identifier to limit search scope.
                    If not provided, searches across all projects.
        limit: Maximum number of results to return (default: 5, max: 20).

    Returns:
        Formatted search results with file paths, content, and relevance info.
    """
    database = _get_database()

    # Only filter by project if explicitly specified
    # None = search across all projects

    # Clamp limit
    limit = min(max(1, limit), 20)

    # Map content_type to DocumentType
    doc_type_filter = None
    if content_type == "code":
        doc_type_filter = DocumentType.CODE
    elif content_type == "documentation":
        doc_type_filter = DocumentType.DOCUMENTATION
    elif content_type == "lesson":
        doc_type_filter = DocumentType.LESSON
    # "all" means no filter

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects
            doc_type=doc_type_filter,
            limit=limit,
        )

        if not results:
            return f"No results found for query: '{query}'" + (
                f" (filtered by {content_type})" if content_type != "all" else ""
            )

        # Format results
        content_label = f" [{content_type.upper()}]" if content_type != "all" else ""
        output_parts = [f"## Search Results{content_label}: '{query}'", ""]

        for i, result in enumerate(results, 1):
            type_badge = f"[{result.doc_type.upper()}]"
            output_parts.append(f"### Result {i}: {type_badge} {result.name}")
            output_parts.append(f"**File:** `{result.file_path}`")
            output_parts.append(f"**Type:** {result.chunk_type} ({result.language})")
            if result.start_line > 0:
                output_parts.append(f"**Lines:** {result.start_line}-{result.end_line}")
            output_parts.append("")
            output_parts.append("```" + result.language)
            output_parts.append(result.text[:2000])  # Truncate long content
            if len(result.text) > 2000:
                output_parts.append("... (truncated)")
            output_parts.append("```")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Search failed: {e!s}"


@mcp.tool()
async def search_docs(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search specifically in documentation (Markdown, RST, text files).

    Use this tool when you need to find information in project documentation,
    README files, or other text documentation. This is more targeted than
    search_knowledge when you know the answer is in the docs.

    Args:
        query: Natural language search query.
               Examples: "how to install", "API configuration", "usage examples".
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Formatted documentation search results.
    """
    database = _get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects
            doc_type=DocumentType.DOCUMENTATION,
            limit=limit,
        )

        if not results:
            return f"No documentation found for: '{query}'"

        output_parts = [f"## Documentation Search: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### {i}. {result.name}")
            output_parts.append(f"**Source:** `{result.file_path}`")
            output_parts.append("")
            # For docs, render as markdown directly
            output_parts.append(result.text[:2500])
            if len(result.text) > 2500:
                output_parts.append("\n... (truncated)")
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Documentation search failed: {e!s}"


@mcp.tool()
async def search_lessons(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search in recorded lessons (problems and solutions).

    Use this tool when you encounter an error or problem that might have been
    solved before. Lessons contain problem descriptions and their solutions,
    making them ideal for troubleshooting similar issues.

    Args:
        query: Description of the problem or error you're facing.
               Examples: "TypeError with None", "database connection timeout",
               "how to fix import error".
        project_id: Optional project identifier. Searches all projects if not specified,
                    enabling cross-project learning.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant lessons with problems and solutions.
    """
    database = _get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects (cross-project learning)
            doc_type=DocumentType.LESSON,
            limit=limit,
        )

        if not results:
            return (
                f"No lessons found matching: '{query}'\n\n"
                "Tip: Use record_lesson to save problems and solutions for future reference."
            )

        output_parts = [f"## Lessons Found: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### Lesson {i}")
            output_parts.append(f"**ID:** {result.name}")
            output_parts.append(f"**Project:** {result.project_id}")
            output_parts.append("")
            output_parts.append(result.text)
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Lesson search failed: {e!s}"


@mcp.tool()
async def search_code(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search specifically in indexed code (functions, classes, methods).

    Use this tool when you need to find code implementations, function definitions,
    or class structures. This is more targeted than search_knowledge when you
    specifically need code, not documentation.

    Args:
        query: Description of the code you're looking for.
               Examples: "function that handles authentication",
               "class for database connections", "method to validate input".
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant code snippets with file locations.
    """
    database = _get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,  # None = all projects
            doc_type=DocumentType.CODE,
            limit=limit,
        )

        if not results:
            return f"No code found for: '{query}'"

        output_parts = [f"## Code Search: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### {i}. {result.chunk_type}: {result.name}")
            output_parts.append(f"**File:** `{result.file_path}`")
            output_parts.append(f"**Lines:** {result.start_line}-{result.end_line}")
            output_parts.append(f"**Language:** {result.language}")
            output_parts.append("")
            output_parts.append("```" + result.language)
            output_parts.append(result.text[:2000])
            if len(result.text) > 2000:
                output_parts.append("... (truncated)")
            output_parts.append("```")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Code search failed: {e!s}"


@mcp.tool()
async def search_tools(
    query: str,
    server: str | None = None,
    limit: int = 5,
) -> str:
    """Search for MCP tools matching a description.

    Use this tool to find other MCP tools when you need to perform an action
    but don't know which tool to use. Returns tool names, descriptions, and
    parameter schemas.

    Args:
        query: Natural language description of what you want to do.
               Examples: "create a GitHub issue", "list files in directory",
               "send a notification to Home Assistant"
        server: Optional server name to filter results (e.g., "github").
        limit: Maximum results to return (default: 5, max: 10).

    Returns:
        Matching tools with server, name, description, and parameters.
    """
    database = _get_database()
    limit = min(max(1, limit), 10)

    # Search for tools
    results = await database.search(
        query=query,
        doc_type=DocumentType.TOOL,
        limit=limit,
    )
    logger.debug("[%s] Searching tools with query='%s'", "nexus-dev", query)
    try:
        logger.debug("[%s] DB Path in use: %s", "nexus-dev", database.config.get_db_path())
    except Exception as e:
        logger.debug("[%s] Could not print DB path: %s", "nexus-dev", e)

    logger.debug("[%s] Results found: %d", "nexus-dev", len(results))
    if results:
        logger.debug("[%s] First result: %s (%s)", "nexus-dev", results[0].name, results[0].score)

    # Filter by server if specified
    if server and results:
        results = [r for r in results if r.server_name == server]

    if not results:
        if server:
            return f"No tools found matching: '{query}' in server: '{server}'"
        return f"No tools found matching: '{query}'"

    # Format output
    output_parts = [f"## MCP Tools matching: '{query}'", ""]

    for i, result in enumerate(results, 1):
        # Parse parameters schema from stored JSON
        params = json.loads(result.parameters_schema) if result.parameters_schema else {}

        output_parts.append(f"### {i}. {result.server_name}.{result.name}")
        output_parts.append(f"**Description:** {result.text}")
        output_parts.append("")
        if params:
            output_parts.append("**Parameters:**")
            output_parts.append("```json")
            output_parts.append(json.dumps(params, indent=2))
            output_parts.append("```")
        output_parts.append("")

    return "\n".join(output_parts)


@mcp.tool()
async def list_servers() -> str:
    """List all configured MCP servers and their status.

    Returns:
        List of MCP servers with connection status.
    """
    mcp_config = _get_mcp_config()
    if not mcp_config:
        return "No MCP config. Run 'nexus-mcp init' first."

    output = ["## MCP Servers", ""]

    active = mcp_config.get_active_servers()
    active_names = {name for name, cfg in mcp_config.servers.items() if cfg in active}

    output.append("### Active")
    if active_names:
        for name in sorted(active_names):
            server = mcp_config.servers[name]
            details = ""
            if server.transport in ("sse", "http"):
                details = f"{server.transport.upper()}: {server.url}"
            else:
                details = f"Command: {server.command} {' '.join(server.args)}"
            output.append(f"- **{name}**: `{details}`")
    else:
        output.append("*No active servers*")

    output.append("")
    output.append("### Disabled")
    disabled = [name for name, server in mcp_config.servers.items() if name not in active_names]
    if disabled:
        for name in sorted(disabled):
            server = mcp_config.servers[name]
            status = "disabled" if not server.enabled else "not in profile"
            output.append(f"- {name} ({status})")
    else:
        output.append("*No disabled servers*")

    return "\n".join(output)


@mcp.tool()
async def get_tool_schema(server: str, tool: str) -> str:
    """Get the full JSON schema for a specific MCP tool.

    Use this after search_tools to get complete parameter details
    before calling invoke_tool.

    Args:
        server: Server name (e.g., "github")
        tool: Tool name (e.g., "create_pull_request")

    Returns:
        Full JSON schema with parameter types and descriptions.
    """
    mcp_config = _get_mcp_config()
    if not mcp_config:
        return "No MCP config. Run 'nexus-mcp init' first."

    if server not in mcp_config.servers:
        available = ", ".join(sorted(mcp_config.servers.keys()))
        return f"Server not found: {server}. Available: {available}"

    server_config = mcp_config.servers[server]
    if not server_config.enabled:
        return f"Server is disabled: {server}"

    conn_manager = _get_connection_manager()

    try:
        connection = await conn_manager.get_connection(server, server_config)
        tools_result = await connection.list_tools()

        for t in tools_result.tools:
            if t.name == tool:
                return json.dumps(
                    {
                        "server": server,
                        "tool": tool,
                        "description": t.description or "",
                        "parameters": t.inputSchema or {},
                    },
                    indent=2,
                )

        available_tools = [t.name for t in tools_result.tools[:10]]
        hint = f" Available: {', '.join(available_tools)}..." if available_tools else ""
        return f"Tool not found: {server}.{tool}.{hint}"

    except Exception as e:
        return f"Error connecting to {server}: {e}"


@mcp.tool()
async def invoke_tool(
    server: str,
    tool: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Invoke a tool on a backend MCP server.

    Use search_tools first to find the right tool, then use this
    to execute it.

    Args:
        server: MCP server name (e.g., "github", "homeassistant")
        tool: Tool name (e.g., "create_issue", "turn_on_light")
        arguments: Tool arguments as dictionary

    Returns:
        Tool execution result.

    Example:
        invoke_tool(
            server="github",
            tool="create_issue",
            arguments={
                "owner": "myorg",
                "repo": "myrepo",
                "title": "Bug fix",
                "body": "Fixed the thing"
            }
        )
    """
    mcp_config = _get_mcp_config()
    if not mcp_config:
        return "No MCP config. Run 'nexus-mcp init' first."

    if server not in mcp_config.servers:
        available = ", ".join(sorted(mcp_config.servers.keys()))
        return f"Server not found: {server}. Available: {available}"

    server_config = mcp_config.servers[server]

    if not server_config.enabled:
        return f"Server is disabled: {server}"

    conn_manager = _get_connection_manager()

    try:
        result = await conn_manager.invoke_tool(
            server,
            server_config,
            tool,
            arguments or {},
        )

        # Format result for AI consumption
        if hasattr(result, "content"):
            # MCP CallToolResult object
            contents = []
            for item in result.content:
                if hasattr(item, "text"):
                    contents.append(item.text)
                else:
                    contents.append(str(item))
            return "\n".join(contents) if contents else "Tool executed successfully (no output)"

        return str(result)

    except Exception as e:
        return f"Tool invocation failed: {e}"


@mcp.tool()
async def index_file(
    file_path: str,
    content: str | None = None,
    project_id: str | None = None,
) -> str:
    """Index a file into the knowledge base.

    Parses the file using language-aware chunking (extracting functions, classes,
    methods) and stores it in the vector database for semantic search.

    Supported file types:
    - Python (.py, .pyw)
    - JavaScript (.js, .jsx, .mjs, .cjs)
    - TypeScript (.ts, .tsx, .mts, .cts)
    - Java (.java)
    - Markdown (.md, .markdown)
    - RST (.rst)
    - Plain text (.txt)

    Args:
        file_path: Path to the file (relative or absolute). The file must exist
                   unless content is provided.
        content: Optional file content. If not provided, reads from disk.
        project_id: Optional project identifier. Uses current project if not specified.

    Returns:
        Summary of indexed chunks including count and types.
    """
    config = _get_config()
    if project_id:
        effective_project_id = project_id
    elif config:
        effective_project_id = config.project_id
    else:
        return (
            "Error: No project_id specified and no nexus_config.json found. "
            "Please provide project_id or run 'nexus-init' first."
        )

    # Resolve file path
    path = Path(file_path)
    if not path.is_absolute():
        path = Path.cwd() / path

    # Get content
    if content is None:
        if not path.exists():
            return f"Error: File not found: {path}"
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e!s}"

    # Determine document type
    doc_type = DocumentType.CODE
    ext = path.suffix.lower()
    if ext in (".md", ".markdown", ".rst", ".txt"):
        doc_type = DocumentType.DOCUMENTATION

    try:
        # Delete existing chunks for this file
        database = _get_database()
        await database.delete_by_file(str(path), effective_project_id)

        # Special handling for lessons to preserve them as single atomic units
        # Check if file is in .nexus/lessons or has lesson frontmatter
        is_lesson = ".nexus/lessons" in str(path) or (
            content.startswith("---") and "problem:" in content[:200]
        )

        if is_lesson:
            doc_type = DocumentType.LESSON
            chunks = [
                CodeChunk(
                    content=content,
                    chunk_type=ChunkType.LESSON,
                    name=path.stem,
                    start_line=1,
                    end_line=content.count("\n") + 1,
                    language="markdown",
                    file_path=str(path),
                )
            ]
        else:
            # Chunk the file normally
            chunks = ChunkerRegistry.chunk_file(path, content)

        if not chunks:
            return f"No indexable content found in: {file_path}"

        # Index chunks
        await _index_chunks(chunks, effective_project_id, doc_type)

        # Summarize by chunk type
        type_counts: dict[str, int] = {}
        for chunk in chunks:
            ctype = chunk.chunk_type.value
            type_counts[ctype] = type_counts.get(ctype, 0) + 1

    except Exception as e:
        return f"Error indexing {path.name}: {e}"

    return f"Indexed {len(chunks)} chunks from {path.name}: {type_counts}"


@mcp.tool()
async def import_github_issues(
    repo: str,
    owner: str,
    limit: int = 10,
    state: str = "all",
) -> str:
    """Import GitHub issues and pull requests into the knowledge base.

    Imports issues from the specified repository using the 'github' MCP server.
    Items are indexed for semantic search (search_knowledge) and can be filtered
    by 'github_issue' or 'github_pr' content types.

    Args:
        repo: Repository name (e.g., "nexus-dev").
        owner: Repository owner (e.g., "mmornati").
        limit: Maximum number of issues to import (default: 10).
        state: Issue state filter: "open" (default), "closed", or "all".

    Returns:
        Summary of imported items.
    """
    database = _get_database()
    config = _get_config()

    if not config:
        return "Error: No project configuration found. Run 'nexus-init' first."

    from .mcp_client import MCPClientManager

    client_manager = MCPClientManager()
    mcp_config = _get_mcp_config()

    importer = GitHubImporter(database, config.project_id, client_manager, mcp_config)

    try:
        count = await importer.import_issues(owner, repo, limit, state)
        return f"Successfully imported {count} issues/PRs from {owner}/{repo}."
    except Exception as e:
        return f"Failed to import issues: {e!s}"


@mcp.tool()
async def record_lesson(
    problem: str,
    solution: str,
    context: str | None = None,
    code_snippet: str | None = None,
    problem_code: str | None = None,
    solution_code: str | None = None,
    project_id: str | None = None,
) -> str:
    """Record a learned lesson from debugging or problem-solving.

    Use this tool to store problems you've encountered and their solutions.
    These lessons will be searchable and can help with similar issues in the future,
    both in this project and across other projects.

    Args:
        problem: Clear description of the problem encountered.
                 Example: "TypeError when passing None to user_service.get_user()"
        solution: How the problem was resolved.
                  Example: "Added null check before calling get_user() and return early if None"
        context: Optional additional context like file path, library, error message.
        code_snippet: Optional code snippet that demonstrates the problem or solution.
                      (Deprecated: use problem_code and solution_code for better structure)
        problem_code: Code snippet showing the problematic code.
        solution_code: Code snippet showing the fixed code.
        project_id: Optional project identifier. Uses current project if not specified.

    Returns:
        Confirmation with lesson ID and a summary.
    """
    import yaml

    config = _get_config()
    if project_id:
        effective_project_id = project_id
    elif config:
        effective_project_id = config.project_id
    else:
        return (
            "Error: No project_id specified and no nexus_config.json found. "
            "Please provide project_id or run 'nexus-init' first."
        )

    # Create lesson text (with YAML frontmatter)
    frontmatter = {
        "problem": problem,
        "timestamp": datetime.now(UTC).isoformat(),
        "project_id": effective_project_id,
        "context": context or "",
        "problem_code": problem_code or "",
        "solution_code": solution_code or "",
    }

    lesson_parts = [
        "---",
        yaml.dump(frontmatter, sort_keys=False).strip(),
        "---",
        "",
        "# Lesson: " + (problem[:50] + "..." if len(problem) > 50 else problem),
        "",
        "## Problem",
        problem,
        "",
        "## Solution",
        solution,
    ]

    if context:
        lesson_parts.extend(["", "## Context", context])

    if problem_code:
        lesson_parts.extend(["", "## Problem Code", "```", problem_code, "```"])

    if solution_code:
        lesson_parts.extend(["", "## Solution Code", "```", solution_code, "```"])

    # Legacy support
    if code_snippet and not (problem_code or solution_code):
        lesson_parts.extend(["", "## Code", "```", code_snippet, "```"])

    lesson_text = "\n".join(lesson_parts)

    # Create a unique ID for this lesson
    lesson_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(UTC).isoformat()

    try:
        embedder = _get_embedder()
        database = _get_database()

        # Generate embedding
        embedding = await embedder.embed(lesson_text)

        # Create document
        doc = Document(
            id=generate_document_id(effective_project_id, "lessons", lesson_id, 0),
            text=lesson_text,
            vector=embedding,
            project_id=effective_project_id,
            file_path=f".nexus/lessons/{lesson_id}.md",
            doc_type=DocumentType.LESSON,
            chunk_type="lesson",
            language="markdown",
            name=f"lesson_{lesson_id}",
            start_line=0,
            end_line=0,
        )

        await database.upsert_document(doc)

        # Also save to .nexus/lessons directory if it exists
        lessons_dir = Path.cwd() / ".nexus" / "lessons"
        if lessons_dir.exists():
            lesson_file = lessons_dir / f"{lesson_id}_{timestamp[:10]}.md"
            try:
                lesson_file.write_text(lesson_text, encoding="utf-8")
            except Exception:
                pass  # Silently fail if we can't write to disk

        return (
            f"✅ Lesson recorded!\n"
            f"- ID: {lesson_id}\n"
            f"- Project: {effective_project_id}\n"
            f"- Problem: {problem[:100]}{'...' if len(problem) > 100 else ''}"
        )

    except Exception as e:
        return f"Failed to record lesson: {e!s}"


@mcp.tool()
async def record_insight(
    category: str,
    description: str,
    reasoning: str,
    correction: str | None = None,
    files_affected: list[str] | None = None,
    project_id: str | None = None,
) -> str:
    """Record an insight from LLM reasoning during development.

    Use this tool to capture:
    - Mistakes made (wrong version, incompatible library, bad approach)
    - Discoveries during exploration (useful patterns, gotchas)
    - Backtracking decisions and their reasons
    - Optimization opportunities found

    Args:
        category: Type of insight - "mistake", "discovery", "backtrack", or "optimization"
        description: What happened (e.g., "Used httpx 0.23 which is incompatible with Python 3.13")
        reasoning: Why it happened / what you were thinking
                   (e.g., "Assumed latest version would work, didn't check compatibility")
        correction: How it was fixed (for mistakes/backtracking)
        files_affected: Optional list of affected file paths
        project_id: Optional project identifier. Uses current project if not specified.

    Returns:
        Confirmation with insight ID and summary.
    """
    import yaml

    config = _get_config()
    if project_id:
        effective_project_id = project_id
    elif config:
        effective_project_id = config.project_id
    else:
        return (
            "Error: No project_id specified and no nexus_config.json found. "
            "Please provide project_id or run 'nexus-init' first."
        )

    # Validate category
    valid_categories = {"mistake", "discovery", "backtrack", "optimization"}
    if category not in valid_categories:
        return f"Error: category must be one of {valid_categories}, got '{category}'"

    # Create insight text with YAML frontmatter
    frontmatter = {
        "category": category,
        "timestamp": datetime.now(UTC).isoformat(),
        "project_id": effective_project_id,
        "files_affected": files_affected or [],
    }

    insight_parts = [
        "---",
        yaml.dump(frontmatter, sort_keys=False).strip(),
        "---",
        "",
        f"# Insight: {category.title()}",
        "",
        "## Description",
        description,
        "",
        "## Reasoning",
        reasoning,
    ]

    if correction:
        insight_parts.extend(["", "## Correction", correction])

    if files_affected:
        insight_parts.extend(["", "## Affected Files", ""])
        for file_path in files_affected:
            insight_parts.append(f"- `{file_path}`")

    insight_text = "\n".join(insight_parts)

    # Create unique ID
    insight_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(UTC).isoformat()

    try:
        embedder = _get_embedder()
        database = _get_database()

        # Generate embedding
        embedding = await embedder.embed(insight_text)

        # Create document
        doc = Document(
            id=generate_document_id(effective_project_id, "insights", insight_id, 0),
            text=insight_text,
            vector=embedding,
            project_id=effective_project_id,
            file_path=f".nexus/insights/{insight_id}.md",
            doc_type=DocumentType.INSIGHT,
            chunk_type="insight",
            language="markdown",
            name=f"{category}_{insight_id}",
            start_line=0,
            end_line=0,
        )

        await database.upsert_document(doc)

        # Save to .nexus/insights directory if it exists
        insights_dir = Path.cwd() / ".nexus" / "insights"
        if insights_dir.exists():
            insight_file = insights_dir / f"{insight_id}_{timestamp[:10]}.md"
            try:
                insight_file.write_text(insight_text, encoding="utf-8")
            except Exception:
                pass

        return (
            f"✅ Insight recorded!\n"
            f"- ID: {insight_id}\n"
            f"- Category: {category}\n"
            f"- Project: {effective_project_id}\n"
            f"- Description: {description[:100]}{'...' if len(description) > 100 else ''}"
        )

    except Exception as e:
        return f"Failed to record insight: {e!s}"


@mcp.tool()
async def search_insights(
    query: str,
    category: str | None = None,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search recorded insights from past development sessions.

    Use when:
    - Starting work on similar features
    - Debugging issues that might have been seen before
    - Looking for optimization patterns
    - Checking if a mistake was already made

    Args:
        query: Description of what you're looking for.
               Examples: "httpx compatibility", "authentication mistakes",
               "database optimization patterns"
        category: Optional filter - "mistake", "discovery", "backtrack", or "optimization"
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant insights with category, description, and reasoning.
    """
    database = _get_database()
    limit = min(max(1, limit), 20)

    # Validate category if provided
    if category:
        valid_categories = {"mistake", "discovery", "backtrack", "optimization"}
        if category not in valid_categories:
            return f"Error: category must be one of {valid_categories}, got '{category}'"

    try:
        results = await database.search(
            query=query,
            project_id=project_id,
            doc_type=DocumentType.INSIGHT,
            limit=limit,
        )

        # Filter by category if specified
        if category and results:
            results = [r for r in results if category in r.name]

        if not results:
            msg = f"No insights found for: '{query}'"
            if category:
                msg += f" (category: {category})"
            return msg + "\n\nTip: Use record_insight to save insights for future reference."

        output_parts = [f"## Insights Found: '{query}'", ""]
        if category:
            output_parts[0] += f" (category: {category})"

        for i, result in enumerate(results, 1):
            output_parts.append(f"### Insight {i}")
            output_parts.append(f"**ID:** {result.name}")
            output_parts.append(f"**Project:** {result.project_id}")
            output_parts.append("")
            output_parts.append(result.text)
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Insight search failed: {e!s}"


@mcp.tool()
async def record_implementation(
    title: str,
    summary: str,
    approach: str,
    design_decisions: list[str],
    files_changed: list[str],
    related_plan: str | None = None,
    project_id: str | None = None,
) -> str:
    """Record a completed implementation for future reference.

    Use this tool after completing a feature or significant work to capture:
    - What was built and why
    - Technical approach used
    - Key design decisions
    - Files involved

    Args:
        title: Short title (e.g., "Add user authentication", "Refactor database layer")
        summary: What was implemented (1-3 sentences)
        approach: How it was done - technical approach/architecture used
        design_decisions: List of key decisions with rationale
                         (e.g., ["Used JWT over sessions for stateless auth",
                                 "Chose Redis for session cache due to speed requirements"])
        files_changed: List of files modified/created
        related_plan: Optional path or URL to implementation plan
        project_id: Optional project identifier. Uses current project if not specified.

    Returns:
        Confirmation with implementation ID and summary.
    """
    import yaml

    config = _get_config()
    if project_id:
        effective_project_id = project_id
    elif config:
        effective_project_id = config.project_id
    else:
        return (
            "Error: No project_id specified and no nexus_config.json found. "
            "Please provide project_id or run 'nexus-init' first."
        )

    # Create implementation text with YAML frontmatter
    frontmatter = {
        "title": title,
        "timestamp": datetime.now(UTC).isoformat(),
        "project_id": effective_project_id,
        "files_changed": files_changed,
        "related_plan": related_plan or "",
    }

    impl_parts = [
        "---",
        yaml.dump(frontmatter, sort_keys=False).strip(),
        "---",
        "",
        f"# Implementation: {title}",
        "",
        "## Summary",
        summary,
        "",
        "## Technical Approach",
        approach,
        "",
        "## Design Decisions",
    ]

    for decision in design_decisions:
        impl_parts.append(f"- {decision}")

    impl_parts.extend(["", "## Files Changed", ""])
    for file_path in files_changed:
        impl_parts.append(f"- `{file_path}`")

    impl_text = "\n".join(impl_parts)

    # Create unique ID
    impl_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now(UTC).isoformat()

    try:
        embedder = _get_embedder()
        database = _get_database()

        # Generate embedding
        embedding = await embedder.embed(impl_text)

        # Create document
        doc = Document(
            id=generate_document_id(effective_project_id, "implementations", impl_id, 0),
            text=impl_text,
            vector=embedding,
            project_id=effective_project_id,
            file_path=f".nexus/implementations/{impl_id}.md",
            doc_type=DocumentType.IMPLEMENTATION,
            chunk_type="implementation",
            language="markdown",
            name=f"impl_{impl_id}",
            start_line=0,
            end_line=0,
        )

        await database.upsert_document(doc)

        # Save to .nexus/implementations directory if it exists
        impl_dir = Path.cwd() / ".nexus" / "implementations"
        if impl_dir.exists():
            impl_file = impl_dir / f"{impl_id}_{timestamp[:10]}.md"
            try:
                impl_file.write_text(impl_text, encoding="utf-8")
            except Exception:
                pass

        return (
            f"✅ Implementation recorded!\n"
            f"- ID: {impl_id}\n"
            f"- Title: {title}\n"
            f"- Project: {effective_project_id}\n"
            f"- Files: {len(files_changed)} changed"
        )

    except Exception as e:
        return f"Failed to record implementation: {e!s}"


@mcp.tool()
async def search_implementations(
    query: str,
    project_id: str | None = None,
    limit: int = 5,
) -> str:
    """Search recorded implementations.

    Use to find:
    - How similar features were built
    - Design patterns used in the project
    - Past technical approaches
    - Implementation history

    Args:
        query: What you're looking for.
               Examples: "authentication implementation", "database refactor",
               "API design patterns"
        project_id: Optional project identifier. Searches all projects if not specified.
        limit: Maximum number of results (default: 5, max: 20).

    Returns:
        Relevant implementations with summaries and design decisions.
    """
    database = _get_database()
    limit = min(max(1, limit), 20)

    try:
        results = await database.search(
            query=query,
            project_id=project_id,
            doc_type=DocumentType.IMPLEMENTATION,
            limit=limit,
        )

        if not results:
            return (
                f"No implementations found for: '{query}'\n\n"
                "Tip: Use record_implementation after completing features to save them."
            )

        output_parts = [f"## Implementations Found: '{query}'", ""]

        for i, result in enumerate(results, 1):
            output_parts.append(f"### Implementation {i}")
            output_parts.append(f"**ID:** {result.name}")
            output_parts.append(f"**Project:** {result.project_id}")
            output_parts.append("")
            # Truncate long content
            output_parts.append(result.text[:3000])
            if len(result.text) > 3000:
                output_parts.append("\n... (truncated)")
            output_parts.append("")
            output_parts.append("---")
            output_parts.append("")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Implementation search failed: {e!s}"


@mcp.resource("mcp://nexus-dev/active-tools")
async def get_active_tools_resource() -> str:
    """List MCP tools from active profile servers.

    Returns a list of tools that are available based on the current
    profile configuration in .nexus/mcp_config.json.
    """
    mcp_config = _get_mcp_config()
    if not mcp_config:
        return "No MCP config found. Run 'nexus-mcp init' first."

    database = _get_database()
    active_servers = _get_active_server_names()

    if not active_servers:
        return f"No active servers in profile: {mcp_config.active_profile}"

    # Query all tools once from the database
    all_tools = await database.search(
        query="",
        doc_type=DocumentType.TOOL,
        limit=1000,  # Get all tools
    )

    # Filter tools by active servers
    tools = [t for t in all_tools if t.server_name in active_servers]

    # Format output
    output = [f"# Active Tools (profile: {mcp_config.active_profile})", ""]

    for server in active_servers:
        server_tools = [t for t in tools if t.server_name == server]
        output.append(f"## {server}")
        if server_tools:
            for tool in server_tools:
                # Truncate description to 100 chars
                desc = tool.text[:100] + "..." if len(tool.text) > 100 else tool.text
                output.append(f"- {tool.name}: {desc}")
        else:
            output.append("*No tools found*")
        output.append("")

    return "\n".join(output)


@mcp.tool()
async def get_project_context(
    project_id: str | None = None,
    limit: int = 10,
) -> str:
    """Get recent lessons and discoveries for a project.

    Returns a summary of recent lessons learned and indexed content for the
    specified project. Useful for getting up to speed on a project or
    reviewing what the AI assistant has learned.

    Args:
        project_id: Project identifier. Uses current project if not specified.
        limit: Maximum number of recent items to return (default: 10).

    Returns:
        Summary of project knowledge including statistics and recent lessons.
    """
    config = _get_config()
    database = _get_database()

    # If no project specified and no config, show stats for all projects
    if project_id is None and config is None:
        project_name = "All Projects"
        effective_project_id = None  # Will get stats for all
    elif project_id is not None:
        project_name = f"Project {project_id[:8]}..."
        effective_project_id = project_id
    else:
        # config is guaranteed not None here (checked at line 595)
        assert config is not None
        project_name = config.project_name
        effective_project_id = config.project_id

    limit = min(max(1, limit), 50)

    try:
        # Get project statistics (None = all projects)
        stats = await database.get_project_stats(effective_project_id)

        # Get recent lessons (None = all projects)
        recent_lessons = await database.get_recent_lessons(effective_project_id, limit)

        # Format output
        output_parts = [
            f"## Project Context: {project_name}",
            f"**Project ID:** `{effective_project_id or 'all'}`",
            "",
            "### Statistics",
            f"- Total indexed chunks: {stats.get('total', 0)}",
            f"- Code chunks: {stats.get('code', 0)}",
            f"- Documentation chunks: {stats.get('documentation', 0)}",
            f"- Lessons: {stats.get('lesson', 0)}",
            "",
        ]

        if recent_lessons:
            output_parts.append("### Recent Lessons")
            output_parts.append("")

            for lesson in recent_lessons:
                import yaml

                output_parts.append(f"#### {lesson.name}")
                # Extract just the problem summary
                # Extract problem from frontmatter or text
                problem = ""
                if lesson.text.startswith("---"):
                    try:
                        # Extract between first and second ---
                        parts = lesson.text.split("---", 2)
                        if len(parts) >= 3:
                            fm = yaml.safe_load(parts[1])
                            problem = fm.get("problem", "")
                    except Exception:
                        pass

                if not problem:
                    lines = lesson.text.split("\n")
                    for i, line in enumerate(lines):
                        if line.strip() == "## Problem" and i + 1 < len(lines):
                            problem = lines[i + 1].strip()
                            break

                if problem:
                    output_parts.append(f"**Problem:** {problem[:200]}...")
                output_parts.append("")

        else:
            output_parts.append("*No lessons recorded yet.*")

        return "\n".join(output_parts)

    except Exception as e:
        return f"Failed to get project context: {e!s}"


async def _get_project_root_from_session(ctx: Context[Any, Any]) -> Path | None:
    """Get the project root from MCP session roots.

    Uses session.list_roots() to query the IDE for workspace folders.

    Args:
        ctx: FastMCP Context with session access.

    Returns:
        Path to the project root if found, None otherwise.
    """
    try:
        # Query the IDE for workspace roots
        roots_result = await ctx.session.list_roots()

        if not roots_result.roots:
            logger.debug("No roots returned from session.list_roots()")
            return None

        # Look for a root that contains nexus_config.json (indicates a nexus project)
        for root in roots_result.roots:
            uri = str(root.uri)
            # Handle file:// URIs
            path = Path(uri[7:]) if uri.startswith("file://") else Path(uri)

            if path.exists() and (path / "nexus_config.json").exists():
                logger.debug("Found nexus project root from session: %s", path)
                return path

        # Fall back to first root if none have nexus_config.json
        first_uri = str(roots_result.roots[0].uri)
        path = Path(first_uri[7:]) if first_uri.startswith("file://") else Path(first_uri)

        if path.exists():
            logger.debug("Using first root from session: %s", path)
            return path

    except Exception as e:
        logger.debug("Failed to get roots from session: %s", e)

    return None


@mcp.tool()
async def list_agents(ctx: Context[Any, Any]) -> str:
    """List available agents in the current workspace.

    Discovers agents from the agents/ directory in the IDE's current workspace.
    Use ask_agent tool to execute tasks with a specific agent.

    Returns:
        List of available agents with names and descriptions.
    """
    # Try to get project root from session (MCP roots)
    project_root = await _get_project_root_from_session(ctx)

    # Fall back to environment variable or cwd
    if not project_root:
        project_root = _find_project_root()

    if not project_root:
        return "No project root found. Make sure you have a nexus_config.json in your workspace."

    agents_dir = project_root / "agents"
    if not agents_dir.exists():
        return f"No agents directory found at {agents_dir}. Create it and add agent YAML files."

    # Load agents from directory
    agent_manager = AgentManager(agents_dir=agents_dir)

    if len(agent_manager) == 0:
        return f"No agents found in {agents_dir}. Add YAML agent configuration files."

    lines = ["# Available Agents", ""]
    for agent in agent_manager:
        lines.append(f"## {agent.display_name or agent.name}")
        lines.append(f"- **Name:** `{agent.name}`")
        lines.append(f"- **Description:** {agent.description}")
        if agent.profile:
            lines.append(f"- **Role:** {agent.profile.role}")
        lines.append("")

    lines.append("Use `ask_agent` tool with the agent name to execute a task.")
    return "\n".join(lines)


@mcp.tool()
async def ask_agent(agent_name: str, task: str, ctx: Context[Any, Any]) -> str:
    """Execute a task using a custom agent from the current workspace.

    Loads the specified agent from the workspace's agents/ directory and
    executes the given task.

    Args:
        agent_name: Name of the agent to use (e.g., 'nexus_architect').
        task: The task description to execute.

    Returns:
        Agent's response.
    """
    # Get database
    database = _get_database()
    if database is None:
        return "Database not initialized. Run nexus-init first."

    # Try to get project root from session (MCP roots)
    project_root = await _get_project_root_from_session(ctx)

    # Fall back to environment variable or cwd
    if not project_root:
        project_root = _find_project_root()

    if not project_root:
        return "No project root found. Make sure you have a nexus_config.json in your workspace."

    agents_dir = project_root / "agents"
    if not agents_dir.exists():
        return f"No agents directory found at {agents_dir}."

    # Load agents from directory
    agent_manager = AgentManager(agents_dir=agents_dir)
    agent_config = agent_manager.get_agent(agent_name)

    if not agent_config:
        available = [a.name for a in agent_manager]
        return f"Agent '{agent_name}' not found. Available agents: {available}"

    # Execute the task
    try:
        executor = AgentExecutor(agent_config, database, mcp)
        config = _get_config()
        project_id = config.project_id if config else None
        return await executor.execute(task, project_id)
    except Exception as e:
        logger.error("Agent execution failed: %s", e, exc_info=True)
        return f"Agent execution failed: {e!s}"


@mcp.tool()
async def refresh_agents(ctx: Context[Any, Any]) -> str:
    """Discovers and registers individual agent tools from the current workspace.

    This tool:
    1. Queries the IDE for the current workspace root.
    2. Scans the 'agents/' directory for agent configurations.
    3. Dynamically registers 'ask_<agent_name>' tools for each agent found.
    4. Notifies the IDE that the tool list has changed.

    Returns:
        A report of registered agents or an error message.
    """
    project_root = await _get_project_root_from_session(ctx)
    if not project_root:
        return "No nexus project root found in workspace (nexus_config.json missing)."

    # Persist the root globally so other tools find it
    global _project_root
    _project_root = project_root

    # Reload other configs if they were initialized lazily from /
    global _config, _mcp_config, _database
    _config = None
    _mcp_config = None
    _database = None

    database = _get_database()
    if database is None:
        return "Database not initialized."

    agents_dir = project_root / "agents"
    if not agents_dir.exists():
        return f"No agents directory found at {agents_dir}."

    global _agent_manager
    _agent_manager = AgentManager(agents_dir=agents_dir)

    if len(_agent_manager) == 0:
        return "No agents found in agents/ directory."

    # Register the tools
    _register_agent_tools(database, _agent_manager)

    # Notify the client that the tool list has changed
    try:
        await ctx.session.send_tool_list_changed()
    except Exception as e:
        logger.warning("Failed to send tool_list_changed notification: %s", e)

    agent_names = [a.name for a in _agent_manager]
    return f"Successfully registered {len(agent_names)} agent tools: {', '.join(agent_names)}"


def _register_agent_tools(database: NexusDatabase, agent_manager: AgentManager | None) -> None:
    """Register dynamic tools for each loaded agent.

    Each agent becomes an MCP tool named `ask_<agent_name>`.
    """
    if agent_manager is None:
        return

    for agent_config in agent_manager:

        def create_agent_tool(cfg: AgentConfig) -> Any:
            """Create a closure to capture the agent config."""

            async def agent_tool(task: str) -> str:
                """Execute a task using the configured agent.

                Args:
                    task: The task description to execute.

                Returns:
                    Agent's response.
                """
                logger.info("Agent tool called: ask_%s for task: %s", cfg.name, task[:100])
                executor = AgentExecutor(cfg, database, mcp)
                config = _get_config()
                project_id = config.project_id if config else None
                return await executor.execute(task, project_id)

            # Set the docstring dynamically
            agent_tool.__doc__ = cfg.description
            return agent_tool

        tool_name = f"ask_{agent_config.name}"
        tool_func = create_agent_tool(agent_config)

        # We use mcp.add_tool directly to allow dynamic registration at runtime
        # FastMCP.tool is a decorator, add_tool is the underlying method
        mcp.add_tool(fn=tool_func, name=tool_name, description=agent_config.description)
        logger.info("Registered agent tool: %s", tool_name)


def main() -> None:
    """Run the MCP server."""
    import argparse
    import signal
    import sys
    from types import FrameType

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Nexus-Dev MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport mode: stdio (default) or sse for Docker/network deployment",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for SSE transport (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    # Configure logging to always use stderr and a debug file
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Stderr handler
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(stderr_handler)

    # File handler for persistent debugging
    try:
        file_handler = logging.FileHandler("/tmp/nexus-dev-debug.log")
        file_handler.setFormatter(logging.Formatter(log_format))
        file_handler.setLevel(logging.DEBUG)
        root_logger.addHandler(file_handler)
    except Exception:
        pass  # Fallback if /tmp is not writable

    root_logger.setLevel(logging.DEBUG)

    # Also ensure the module-specific logger is at INFO
    logger.setLevel(logging.DEBUG)

    def handle_signal(sig: int, frame: FrameType | None) -> None:
        logger.info("Received signal %s, shutting down...", sig)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # Initialize on startup
    try:
        logger.info("Starting Nexus-Dev MCP server...")
        _get_config()
        database = _get_database()
        _get_mcp_config()

        # Load and register custom agents
        # Find project root and look for agents directory
        logger.debug("Current working directory: %s", Path.cwd())
        project_root = _find_project_root()
        agents_dir = project_root / "agents" if project_root else None
        logger.debug("Project root: %s", project_root)
        logger.debug("Agents directory: %s", agents_dir)

        global _agent_manager
        _agent_manager = AgentManager(agents_dir=agents_dir)
        _register_agent_tools(database, _agent_manager)

        # Run server with selected transport
        if args.transport == "sse":
            logger.info(
                "Server initialization complete, running SSE transport on %s:%d",
                args.host,
                args.port,
            )
            mcp.run(transport="sse", host=args.host, port=args.port)  # type: ignore
        else:
            logger.info("Server initialization complete, running stdio transport")
            mcp.run(transport="stdio")
    except Exception as e:
        logger.critical("Fatal error in MCP server: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        logger.info("MCP server shutdown complete")


if __name__ == "__main__":
    main()
