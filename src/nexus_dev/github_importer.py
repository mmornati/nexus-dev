"""GitHub Knowledge Importer.

Imports issues and pull requests from GitHub via the MCP server
and indexes them into the Nexus knowledge base.
"""

from __future__ import annotations

import logging
from typing import Any

from .database import Document, DocumentType, NexusDatabase, generate_document_id
from .mcp_client import MCPClientManager, MCPServerConnection
from .mcp_config import MCPConfig

logger = logging.getLogger(__name__)


class GitHubImporter:
    """Imports GitHub data into Nexus knowledge base."""

    def __init__(
        self,
        database: NexusDatabase,
        project_id: str,
        client_manager: MCPClientManager | None = None,
        mcp_config: MCPConfig | None = None,
    ) -> None:
        """Initialize importer.

        Args:
            database: Database instance to store documents.
            project_id: Project ID to associate documents with.
            client_manager: Optional MCP client manager (uses default if None).
            mcp_config: MCP configuration (required to find 'github' server).
        """
        self.database = database
        self.project_id = project_id
        self.client_manager = client_manager or MCPClientManager()
        self.mcp_config = mcp_config

    async def import_issues(
        self,
        owner: str,
        repo: str,
        limit: int = 20,
        state: str = "all",
    ) -> int:
        """Import GitHub issues.

        Args:
            owner: Repository owner.
            repo: Repository name.
            limit: Maximum number of issues to import.
            state: Issue state ("open", "closed", "all").

        Returns:
            Number of issues imported.
        """
        # Connect to GitHub MCP server
        try:
            if not self.mcp_config:
                raise ValueError("MCP Config is required to find 'github' server.")

            server_name = "github"
            server_config = self.mcp_config.servers.get(server_name)
            if not server_config:
                raise ValueError("Server 'github' not found in MCP config.")

            # Create connection object
            connection = MCPServerConnection(
                name=server_name,
                command=server_config.command or "",
                args=server_config.args,
                env=server_config.env,
                transport=server_config.transport,
                url=server_config.url,
                headers=server_config.headers,
                timeout=server_config.timeout,
            )

            all_items = []

            # 1. Fetch Issues
            issues = await self._fetch_tool_items(
                connection, "list_issues", owner, repo, limit, state
            )
            logger.info(f"Fetched {len(issues)} issues")
            all_items.extend(issues)

            # 2. Fetch Pull Requests
            prs = await self._fetch_tool_items(
                connection, "list_pull_requests", owner, repo, limit, state
            )
            logger.info(f"Fetched {len(prs)} PRs")
            all_items.extend(prs)

            if not all_items:
                return 0

            return await self._index_issues(all_items, owner, repo)

        except Exception as e:
            logger.error("Failed to import GitHub data: %s", e)
            raise

    async def _fetch_tool_items(
        self,
        connection: MCPServerConnection,
        tool_name: str,
        owner: str,
        repo: str,
        limit: int,
        state: str,
    ) -> list[dict[str, Any]]:
        """Fetch items using a specific GitHub MCP tool."""
        args = {
            "owner": owner,
            "repo": repo,
            "state": state,
            "per_page": limit,
        }

        try:
            result = await self.client_manager.call_tool(connection, tool_name, args)
        except Exception as e:
            logger.warning("Failed to call tool %s: %s", tool_name, e)
            return []

        if not result:
            return []

        # Extract content
        content_list = []
        if hasattr(result, "content"):
            content_list = result.content
        elif isinstance(result, dict) and "content" in result:
            content_list = result["content"]

        text_content = ""
        for content in content_list:
            if hasattr(content, "text"):
                text_content += content.text
            elif isinstance(content, dict) and "text" in content:
                text_content += content["text"]

        if not text_content:
            return []

        import json

        try:
            items = json.loads(text_content)
        except json.JSONDecodeError:
            logger.error("Failed to parse %s response as JSON", tool_name)
            return []

        if not isinstance(items, list):
            # Handle dictionary response
            if isinstance(items, dict):
                # Check for common wrapper keys
                for key in ["items", "issues", "pull_requests", "data"]:
                    if key in items:
                        items = items[key]
                        break

            if items is None:
                items = []
            elif not isinstance(items, list):
                # Fallback: if it's a dict but we couldn't find a list, maybe it's a single item?
                # Or unexpected format.
                items = [items] if isinstance(items, dict) and items else []

        return items

    async def _index_issues(self, issues: list[dict[str, Any]], owner: str, repo: str) -> int:
        """Index a list of issue dictionaries."""
        docs = []
        embedder = self.database.embedder

        # Batch embed would be efficient, but let's do one by one or prepare list first
        # We need text for embedding
        texts_to_embed = []
        valid_issues = []

        for issue in issues:
            # Skip pull requests if they come through list_issues API (they often do)
            # Unless we want to index them too (DocumentType.GITHUB_PR)

            # GitHub API: PRs are issues. They have a 'pull_request' key.
            # But from list_pull_requests endpoint, they have 'base' key.
            is_pr = "pull_request" in issue or "base" in issue
            doc_type = DocumentType.GITHUB_PR if is_pr else DocumentType.GITHUB_ISSUE

            number = issue.get("number")
            title = issue.get("title", "")
            body = issue.get("body", "") or ""
            url = issue.get("html_url", "")
            # state = issue.get("state", "")

            # Create a rich text representation for RAG
            text = f"""GitHub {doc_type.value.replace("_", " ").title()} #{number}: {title}
Repo: {owner}/{repo}
URL: {url}
State: {issue.get("state")}
Author: {issue.get("user", {}).get("login")}

{body}
"""
            texts_to_embed.append(text)
            valid_issues.append((issue, doc_type, text, number, url, title))

        if not texts_to_embed:
            return 0

        # Generate embeddings
        # Assuming embedder has embed_batch
        embeddings = await embedder.embed_batch(texts_to_embed)

        for (_, doc_type, text, number, url, title), vector in zip(
            valid_issues, embeddings, strict=True
        ):
            doc_id = generate_document_id(
                self.project_id, f"github://{owner}/{repo}/issues/{number}", str(number), 0
            )

            # Metadata to store
            # We can use 'name' for Issue #123
            name = f"Issue #{number}: {title}"

            doc = Document(
                id=doc_id,
                text=text,
                vector=vector,
                project_id=self.project_id,
                file_path=url,  # Use URL as file_path for clickable links
                doc_type=doc_type,
                chunk_type="issue" if doc_type == DocumentType.GITHUB_ISSUE else "pr",
                language="markdown",
                name=name,
                start_line=0,
                end_line=0,
            )
            docs.append(doc)

        await self.database.upsert_documents(docs)
        return len(docs)
