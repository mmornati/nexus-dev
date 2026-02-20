"""Indexing commands."""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path

import click

from nexus_dev.chunkers import ChunkerRegistry
from nexus_dev.cli.utils import (
    _detect_document_type_and_metadata,
    _index_chunks_sync,
    _is_excluded,
    _print_file_summary,
    _run_async,
    _should_index,
    requires_config,
)
from nexus_dev.config import NexusConfig
from nexus_dev.database import Document, DocumentType, NexusDatabase, generate_document_id
from nexus_dev.embeddings import create_embedder


@click.command("index")
@click.argument("paths", nargs=-1, required=True)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Index directories recursively",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress output",
)
@requires_config
def index_command(
    config: NexusConfig, paths: tuple[str, ...], recursive: bool, quiet: bool
) -> None:
    """Manually index files or directories.

    PATHS can be files or directories. Use -r to recursively index directories.

    Examples:
        nexus-index src/
        nexus-index docs/ -r
        nexus-index main.py utils.py
    """
    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    # Collect files to index
    files_to_index: list[Path] = []

    for path_str in paths:
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.exists():
            if not quiet:
                click.echo(f"⚠️  Path not found: {path_str}")
            continue

        if path.is_file():
            files_to_index.append(path)
        elif path.is_dir():
            if recursive:
                # Recursively find files using os.walk to prune ignored directories
                for root, dirs, files in os.walk(path):
                    root_path = Path(root)

                    # Compute relative path for pattern matching
                    rel_root = str(root_path.relative_to(Path.cwd()))
                    if rel_root == ".":
                        rel_root = ""

                    # Filter directories to prevent traversal into ignored paths
                    # We must modify dirs in-place to prune the walk
                    i = 0
                    while i < len(dirs):
                        d = dirs[i]
                        d_path = root_path / d
                        # We construct a mock path string for the directory check
                        # (relative path + directory name)
                        check_path = str(d_path.relative_to(Path.cwd()))

                        # Use a simpler check: if the directory ITSELF matches exclude pattern
                        # we should remove it.
                        should_exclude = False

                        # Check excludes for this directory
                        # We treat the directory string as match target for exclude patterns
                        # excluding the trailing slash for fnmatch
                        for pattern in config.exclude_patterns:
                            # Normalize pattern: remove trailing slash for directory matching
                            clean_pat = pattern.rstrip("/")
                            if clean_pat.endswith("/**"):
                                clean_pat = clean_pat[:-3]

                            # Simple fnmatch on the logic
                            if fnmatch(check_path, pattern) or fnmatch(check_path, clean_pat):
                                should_exclude = True
                                break

                            # Handle recursive wildcard start (e.g. **/node_modules)
                            if clean_pat.startswith("**/"):
                                suffix = clean_pat[3:]
                                if (
                                    check_path == suffix
                                    or check_path.endswith("/" + suffix)
                                    or fnmatch(check_path, suffix)
                                ):
                                    should_exclude = True
                                    break

                        if should_exclude:
                            if not quiet:
                                # Optional: debug output if needed, but keeping it clean for now
                                pass
                            del dirs[i]
                        else:
                            i += 1

                    # Add files
                    for file in files:
                        file_path = root_path / file
                        if _should_index(file_path, config):
                            files_to_index.append(file_path)
            else:
                # Only immediate children
                for file_path in path.iterdir():
                    if file_path.is_file():
                        # For explicit paths/directories, we check excludes but ignore
                        # include patterns to allow indexing "anything I point at"
                        # unless specifically excluded
                        is_excluded = _is_excluded(file_path, config)
                        if not is_excluded:
                            files_to_index.append(file_path)

    if not files_to_index:
        if not quiet:
            click.echo("No files to index.")
        return

    if not quiet:
        _print_file_summary(files_to_index)
        if not click.confirm("Proceed with indexing?"):
            click.echo("Aborted.")
            return

    # Index files
    total_chunks = 0
    errors = 0

    for file_path in files_to_index:
        try:
            # Read file
            content = file_path.read_text(encoding="utf-8")

            # Detect smart type from frontmatter
            detected_type, metadata = _detect_document_type_and_metadata(content)

            # Determine type
            if detected_type:
                doc_type = detected_type
            else:
                ext = file_path.suffix.lower()
                doc_type = (
                    DocumentType.DOCUMENTATION
                    if ext in (".md", ".markdown", ".rst", ".txt")
                    else DocumentType.CODE
                )

            # Delete existing
            _run_async(database.delete_by_file(str(file_path), config.project_id))

            # Chunk file
            chunks = ChunkerRegistry.chunk_file(file_path, content)

            if chunks:
                # Generate embeddings and store
                chunk_count = _run_async(
                    _index_chunks_sync(
                        chunks,
                        config.project_id,
                        doc_type,
                        embedder,
                        database,
                        metadata=metadata,
                    )
                )
                total_chunks += chunk_count

                if not quiet:
                    click.echo(f"  ✅ {file_path.name}: {chunk_count} chunks")

        except Exception as e:
            errors += 1
            if not quiet:
                click.echo(f"  ❌ {file_path.name}: {e!s}")

    if not quiet:
        click.echo("")
        click.echo(f"✅ Indexed {total_chunks} chunks from {len(files_to_index) - errors} files")
        if errors:
            click.echo(f"⚠️  {errors} file(s) failed")

        # Graph indexing (Python + JS/TS)
    if config.enable_hybrid_db:
        from nexus_dev.code_graph import PythonGraphBuilder
        from nexus_dev.code_graph_js import JSGraphBuilder
        from nexus_dev.hybrid_db import HybridDatabase

        # Define supported extensions and their builders
        graph_files = []
        for f in files_to_index:
            ext = f.suffix.lower()
            if ext in (".py", ".pyw"):
                graph_files.append((f, "python"))
            elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                graph_files.append((f, "js"))

        if not quiet:
            click.echo(f"  (Debug) Hybrid DB enabled: {config.enable_hybrid_db}")
            click.echo(f"  (Debug) Code graph files found: {len(graph_files)}")

        if graph_files:
            if not quiet:
                click.echo("")
                click.echo("🔗 Indexing code graph...")

            try:
                hybrid_db = HybridDatabase(config)
                hybrid_db.connect()

                # Initialize builders
                py_builder = PythonGraphBuilder(hybrid_db.graph, config.project_id)
                js_builder = JSGraphBuilder(hybrid_db.graph, config.project_id)

                graph_errors = 0
                for file_path, lang in graph_files:
                    try:
                        builder = py_builder if lang == "python" else js_builder
                        stats = builder.index_file(file_path)
                        if not quiet:
                            # Format stats string dynamically
                            stats_str = ", ".join(
                                f"{v}{k[0].upper()}" for k, v in stats.items() if v > 0
                            )
                            click.echo(f"  🔗 {file_path.name}: {stats_str or 'No entities'}")
                    except Exception as e:
                        graph_errors += 1
                        if not quiet:
                            click.echo(f"  ⚠️ {file_path.name}: Graph failed - {e}")

                hybrid_db.close()

                if not quiet:
                    click.echo(f"✅ Graph indexed {len(graph_files) - graph_errors} files")
            except Exception as e:
                if not quiet:
                    click.echo(f"⚠️ Graph indexing failed: {e}")


@click.command("index-lesson")
@click.argument("lesson_file")
@click.option("-q", "--quiet", is_flag=True, help="Suppress output")
@requires_config
def index_lesson_command(config: NexusConfig, lesson_file: str, quiet: bool) -> None:
    """Index a lesson file from .nexus/lessons/."""
    path = Path(lesson_file)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        if not quiet:
            click.echo(f"❌ Lesson file not found: {lesson_file}", err=True)
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    try:
        content = path.read_text(encoding="utf-8")

        # Generate embedding
        embedding = _run_async(embedder.embed(content))

        # Create document
        doc_id = generate_document_id(
            config.project_id,
            str(path),
            path.stem,
            0,
        )

        doc = Document(
            id=doc_id,
            text=content,
            vector=embedding,
            project_id=config.project_id,
            file_path=str(path),
            doc_type=DocumentType.LESSON,
            chunk_type="lesson",
            language="markdown",
            name=path.stem,
            start_line=0,
            end_line=0,
        )

        _run_async(database.upsert_document(doc))

        if not quiet:
            click.echo(f"✅ Indexed lesson: {path.name}")

    except Exception as e:
        if not quiet:
            click.echo(f"❌ Failed to index lesson: {e!s}", err=True)


@click.command("clean")
@click.option("--project-id", help="Project ID to clean (default: current project)")
@click.option("--all", "clean_all", is_flag=True, help="Delete ALL projects (dangerous!)")
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting")
def clean_command(project_id: str | None, clean_all: bool, dry_run: bool) -> None:
    """Delete indexed data for a project."""
    # Validate options
    from nexus_dev.cli.utils import _validate_embeddings_or_exit

    if clean_all and project_id:
        click.echo("❌ Cannot use both --all and --project-id", err=True)
        return

    if not clean_all and not project_id:
        # Try to get project_id from current directory
        config_path = Path.cwd() / "nexus_config.json"
        if config_path.exists():
            config = NexusConfig.load(config_path)
            project_id = config.project_id
        else:
            click.echo("❌ No project-id specified and no nexus_config.json found", err=True)
            click.echo("   Use --project-id or run from a project directory", err=True)
            return

    # Load database
    config_path = Path.cwd() / "nexus_config.json"
    if config_path.exists():
        config = NexusConfig.load(config_path)
    else:
        config = NexusConfig.create_new("temp")

    # Validate embedding configuration before proceeding
    if not _validate_embeddings_or_exit(config):
        return

    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    try:
        if clean_all:
            # Get total count
            stats = _run_async(database.get_project_stats(None))
            total = stats.get("total", 0)

            if total == 0:
                click.echo("⚠️  Database is already empty")
                return

            click.echo(f"⚠️  WARNING: This will delete ALL {total} documents from the database!")
            click.echo("")

            if dry_run:
                click.echo("[DRY RUN] Would delete entire database")
                return

            if not click.confirm("Are you absolutely sure?"):
                click.echo("Aborted.")
                return

            # Delete all by resetting
            database.reset()
            click.echo(f"✅ Deleted all {total} documents")

        else:
            # Delete specific project
            stats = _run_async(database.get_project_stats(project_id))
            count = stats.get("total", 0)

            if count == 0:
                click.echo(f"⚠️  No documents found for project: {project_id}")
                return

            click.echo(f"Found {count} documents for project: {project_id}")
            click.echo("")
            click.echo("Document types:")
            for doc_type, type_count in stats.items():
                if doc_type != "total":
                    click.echo(f"  - {doc_type}: {type_count}")
            click.echo("")

            if dry_run:
                click.echo(f"[DRY RUN] Would delete {count} documents for project {project_id}")
                return

            if not click.confirm(f"Delete {count} documents?"):
                click.echo("Aborted.")
                return

            # project_id is guaranteed to be set by validation logic above
            assert project_id is not None
            deleted = _run_async(database.delete_by_project(project_id))
            click.echo(f"✅ Deleted {deleted} documents for project {project_id}")

    except Exception as e:
        click.echo(f"❌ Error during cleanup: {e!s}", err=True)


@click.command("reindex")
@requires_config
def reindex_command(config: NexusConfig) -> None:
    """Re-index entire project (clear and rebuild)."""
    # Collect files first to show summary
    click.echo("🔍 Scanning files...")

    cwd = Path.cwd()
    files_to_index: list[Path] = []

    for pattern in config.include_patterns:
        for file_path in cwd.glob(pattern):
            if file_path.is_file() and _should_index(file_path, config):
                files_to_index.append(file_path)

    # Also index docs folders
    for docs_folder in config.docs_folders:
        docs_path = cwd / docs_folder
        if docs_path.is_file():
            files_to_index.append(docs_path)
        elif docs_path.is_dir():
            for root, _, files in os.walk(docs_path):
                # Apply same pruning logic for docs if needed, though usually docs are safer
                # For consistency let's just collect files
                for file in files:
                    files_to_index.append(Path(root) / file)

    # Remove duplicates
    files_to_index = list(set(files_to_index))

    # Show summary and ask for confirmation
    _print_file_summary(files_to_index)

    if not click.confirm("This will CLEAR the database and re-index the above files. Continue?"):
        click.echo("Aborted.")
        return

    # Proceed with DB operations
    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    click.echo("🗑️  Clearing existing index...")
    # Reset database to handle schema changes
    database.reset()
    # Re-connect to create new table with updated schema
    database.connect()
    click.echo("   Index cleared and schema updated")

    click.echo("")
    click.echo("📁 Re-indexing project...")

    # Index all files
    total_chunks = 0
    for file_path in files_to_index:
        try:
            content = file_path.read_text(encoding="utf-8")
            ext = file_path.suffix.lower()
            doc_type = (
                DocumentType.DOCUMENTATION
                if ext in (".md", ".markdown", ".rst", ".txt")
                else DocumentType.CODE
            )

            chunks = ChunkerRegistry.chunk_file(file_path, content)
            if chunks:
                count = _run_async(
                    _index_chunks_sync(chunks, config.project_id, doc_type, embedder, database)
                )
                total_chunks += count
                click.echo(f"  ✅ {file_path.name}: {count} chunks")

        except Exception as e:
            click.echo(f"  ❌ Failed to index {file_path.name}: {e!s}", err=True)

    click.echo("")
    click.echo(f"✅ Re-indexed {total_chunks} chunks from {len(files_to_index)} files")
