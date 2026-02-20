"""Knowledge management tools for Nexus-Dev."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from nexus_dev.app_state import (
    find_project_root,
    get_config,
    get_database,
    get_embedder,
)
from nexus_dev.chunkers import ChunkerRegistry, ChunkType, CodeChunk
from nexus_dev.database import Document, DocumentType, generate_document_id

logger = logging.getLogger(__name__)


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

    embedder = get_embedder()
    database = get_database()

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
    config = get_config()
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
    root = (find_project_root() or Path.cwd()).resolve()
    path = Path(file_path)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()

    # Security check: ensure path is within root
    try:
        path.relative_to(root)
    except ValueError:
        return f"Error: Access denied. Path is outside project root: {file_path}"

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
        database = get_database()
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

    config = get_config()
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
        embedder = get_embedder()
        database = get_database()

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

    config = get_config()
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
        embedder = get_embedder()
        database = get_database()

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

    config = get_config()
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
        embedder = get_embedder()
        database = get_database()

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
    config = get_config()
    database = get_database()

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
