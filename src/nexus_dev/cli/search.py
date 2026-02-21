"""Search and inspection commands."""

from __future__ import annotations

from pathlib import Path

import click

from nexus_dev.cli.utils import (
    _run_async,
    _validate_embeddings_or_exit,
    requires_config,
)
from nexus_dev.config import NexusConfig
from nexus_dev.database import DocumentType, NexusDatabase
from nexus_dev.embeddings import create_embedder


@click.command("search")
@click.argument("query")
@click.option("--type", "content_type", help="Content type to filter by")
@click.option("--limit", default=5, help="Number of results")
@requires_config
def search_command(config: NexusConfig, query: str, content_type: str | None, limit: int) -> None:
    """Search the knowledge base."""
    embedder = create_embedder(config)
    database = NexusDatabase(config, embedder)
    database.connect()

    click.echo(f"🔍 Searching for '{query}'...")

    # Initialize Hybrid DB (if enabled)
    hybrid_db = None
    if config.enable_hybrid_db:
        from nexus_dev.hybrid_db import HybridDatabase

        hybrid_db = HybridDatabase(config)

    # Smart Search Routing
    from nexus_dev.query_router import HybridQueryRouter, QueryType

    router = HybridQueryRouter()
    intent = router.route(query)

    # 1. Graph Intent
    if (
        intent.query_type == QueryType.GRAPH
        and intent.extracted_entity
        and hybrid_db
        and config.enable_hybrid_db
    ):
        click.echo(f"🧠 Smart Search identified GRAPH intent (Confidence: {intent.confidence})")
        entity = intent.extracted_entity
        q_lower = query.lower()

        try:
            # Check Graph Patterns (logic mirrored from server.py)
            hybrid_db.connect()

            if "calls" in q_lower or "callers" in q_lower:
                click.echo(f"   Finding callers of: {entity}")
                # Find callers logic
                cypher = f"""
                    MATCH call_path = (caller)-[:CALLS]->(callee {{name: $entity}})
                    RETURN caller.name as caller_name, caller.file_path as file,
                           caller.start_line as line
                    ORDER BY file, line
                    LIMIT {limit}
                """
                res = hybrid_db.graph.query(cypher, {"entity": entity})
                if res.result_set:
                    click.echo("\n## Callers:\n")
                    for row in res.result_set:
                        click.echo(f"- {row[0]} (in {row[1]}:{row[2]})")
                    return
                else:
                    click.echo("   No callers found.")

            elif "imports" in q_lower or "dependencies" in q_lower:
                # Default to 'both' unless direction is clear
                direction = "both"
                if "what imports" in q_lower or "who imports" in q_lower:
                    direction = "imported_by"
                elif "what does" in q_lower and "import" in q_lower:
                    direction = "imports"

                click.echo(f"   Searching dependencies for: {entity} (direction: {direction})")

                # Check imports (what target imports)
                if direction in ("imports", "both"):
                    cypher = f"""
                        MATCH import_path = (f:File {{path: $target}})-[:IMPORTS]->(dep:File)
                        RETURN dep.path AS dependency
                        LIMIT {limit}
                    """
                    # Note: Using entity as path might need full path resolution,
                    # but mimicking server logic for now which takes 'target'
                    # In CLI we might need better path resolution if user types just filename
                    # attempting simple match
                    res = hybrid_db.graph.query(cypher, {"target": entity})
                    if res.result_set:
                        click.echo("\n## Imports (depends on):\n")
                        for row in res.result_set:
                            click.echo(f"→ {row[0]}")

                # Check imported by (what imports target)
                if direction in ("imported_by", "both"):
                    cypher = f"""
                        MATCH import_path = (f:File)-[:IMPORTS]->(target:File {{path: $target}})
                        RETURN f.path AS importer
                        LIMIT {limit}
                    """
                    res = hybrid_db.graph.query(cypher, {"target": entity})
                    if res.result_set:
                        click.echo("\n## Imported By (required by):\n")
                        for row in res.result_set:
                            click.echo(f"← {row[0]}")
                return

            elif "implements" in q_lower or "extends" in q_lower or "subclasses" in q_lower:
                click.echo(f"   Finding implementations of: {entity}")
                cypher = f"""
                    MATCH (impl)-[:INHERITS_FROM]->(base {{name: $entity}})
                    RETURN impl.name as class_name, impl.file_path as file
                    LIMIT {limit}
                """
                res = hybrid_db.graph.query(cypher, {"entity": entity})
                if res.result_set:
                    click.echo("\n## Implementations/Subclasses:\n")
                    for row in res.result_set:
                        click.echo(f"↓ {row[0]} (in {row[1]})")
                    return
                else:
                    click.echo("   No implementations found.")

        except Exception as e:
            click.echo(f"⚠️  Graph search failed: {e}")
            click.echo("   Falling back to vector search...")

    # 2. KV Intent (Session Context)
    elif intent.query_type == QueryType.KV:
        click.echo(
            "⚠️  Session context search (KV) is not available in CLI mode (requires active session)."
        )
        click.echo("   Falling back to vector search...")

    doc_type_enum = None
    if content_type:
        try:
            doc_type_enum = DocumentType(content_type)
        except ValueError:
            click.echo(f"⚠️  Invalid type '{content_type}'. Ignoring filter.")

    results = _run_async(database.search(query, limit=limit, doc_type=doc_type_enum))

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nFound {len(results)} results:\n")

    for i, doc in enumerate(results, 1):
        click.echo(f"{i}. [{doc.doc_type.upper()}] {doc.name} (Score: {doc.score:.3f})")
        click.echo(f"   path: {doc.file_path}")
        # Preview text
        text = doc.text.replace("\n", " ").strip()
        if len(text) > 100:
            text = text[:97] + "..."
        click.echo(f'   "{text}"')
        click.echo("")


@click.command("export")
@click.option("--project-id", help="Project ID to export (defaults to current config)")
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output directory (default: ./nexus-export)",
)
def export_command(project_id: str | None, output: Path | None) -> None:
    """Export project knowledge to markdown files."""

    async def _export() -> None:
        # Load config
        config = None
        try:
            config_path = Path.cwd() / "nexus_config.json"
            if config_path.exists():
                config = NexusConfig.load(config_path)
        except Exception:
            pass

        effective_project_id = project_id
        if not effective_project_id and config:
            effective_project_id = config.project_id

        if not effective_project_id:
            click.secho("Error: No project-id provided and no nexus_config.json found.", fg="red")
            return

        # Initialize DB
        if not config:
            # Create temporary config for DB access
            config = NexusConfig.create_new("temp")

        try:
            # Validate embedding configuration before proceeding
            if not _validate_embeddings_or_exit(config):
                return

            embedder = create_embedder(config)
            db = NexusDatabase(config, embedder)
            db.connect()

            click.echo(f"Exporting knowledge for project: {effective_project_id}")

            # Get all documents for this project
            # searching with empty query returns all items for project/type
            # We fetch strictly structured data types: Lesson, Insight, Implementation
            types_to_export = [
                (DocumentType.LESSON, "lessons"),
                (DocumentType.INSIGHT, "insights"),
                (DocumentType.IMPLEMENTATION, "implementations"),
            ]

            base_dir = output or Path.cwd() / "nexus-export"
            base_dir.mkdir(parents=True, exist_ok=True)

            total_count = 0

            for doc_type, dirname in types_to_export:
                # Retrieve all documents of the specific type for the project
                results = await db.list_documents(
                    project_id=effective_project_id,
                    doc_type=doc_type,
                    limit=1000,
                )

                if not results:
                    continue

                type_dir = base_dir / dirname
                type_dir.mkdir(exist_ok=True)

                click.echo(f"  - Found {len(results)} {dirname}")

                for res in results:
                    # Use ID from metadata if available, else generate safe name
                    safe_name = "".join(c for c in res.name if c.isalnum() or c in "-_")
                    filename = f"{safe_name}.md"

                    file_path = type_dir / filename
                    file_path.write_text(res.text, encoding="utf-8")
                    total_count += 1

            click.secho(f"\nSuccessfully exported {total_count} files to {base_dir}", fg="green")

        except Exception as e:
            click.secho(f"Export failed: {e}", fg="red")

    _run_async(_export())


@click.command("status")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed debug information")
@requires_config
def status_command(config: NexusConfig, verbose: bool) -> None:
    """Show Nexus-Dev status and statistics."""
    click.echo("📊 Nexus-Dev Status")
    click.echo("")
    click.echo(f"Project: {config.project_name}")
    click.echo(f"Project ID: {config.project_id}")
    click.echo(f"Embedding Provider: {config.embedding_provider}")
    click.echo(f"Embedding Model: {config.embedding_model}")
    click.echo(f"Database: {config.get_db_path()}")
    click.echo("")

    try:
        embedder = create_embedder(config)
        database = NexusDatabase(config, embedder)
        database.connect()

        if verbose:
            click.echo("🔍 Debug Info:")
            click.echo(f"   Database path exists: {config.get_db_path().exists()}")
            click.echo(f"   Querying for project_id: {config.project_id}")
            click.echo("")

        stats = _run_async(database.get_project_stats(config.project_id))

        click.echo("📈 Statistics:")
        click.echo(f"   Total chunks: {stats.get('total', 0)}")
        click.echo(f"   Code: {stats.get('code', 0)}")
        click.echo(f"   Documentation: {stats.get('documentation', 0)}")
        click.echo(f"   Lessons: {stats.get('lesson', 0)}")

        if verbose and stats.get("total", 0) > 0:
            click.echo("")
            click.echo("   Document type breakdown:")
            for doc_type, count in stats.items():
                if doc_type != "total":
                    click.echo(f"     - {doc_type}: {count}")

    except Exception as e:
        click.echo(f"⚠️  Could not connect to database: {e!s}")
        if verbose:
            import traceback

            click.echo("")
            click.echo("Full traceback:")
            click.echo(traceback.format_exc())


@click.command("inspect")
@click.option("--project-id", help="Filter by project ID (default: current project)")
@click.option("--limit", default=5, help="Number of sample documents to show")
@click.option("--all-projects", is_flag=True, help="Show all projects in database")
def inspect_command(project_id: str | None, limit: int, all_projects: bool) -> None:
    """Inspect database contents for debugging."""
    # Load config for default project_id
    config = None
    if not all_projects and not project_id:
        config_path = Path.cwd() / "nexus_config.json"
        if config_path.exists():
            config = NexusConfig.load(config_path)
            project_id = config.project_id

    # Get database path from config or use default
    if config:
        # Validate embedding configuration before proceeding
        if not _validate_embeddings_or_exit(config):
            return

        embedder = create_embedder(config)
        database = NexusDatabase(config, embedder)
    else:
        # Use default config to access shared database
        default_config = NexusConfig.create_new("temp")
        # Validate embedding configuration before proceeding
        if not _validate_embeddings_or_exit(default_config):
            return

        embedder = create_embedder(default_config)
        database = NexusDatabase(default_config, embedder)

    database.connect()

    click.echo("🔍 Nexus-Dev Database Inspection")
    click.echo("")

    try:
        # Get database info
        db_path = database.config.get_db_path()
        click.echo(f"Database location: {db_path}")

        if db_path.exists():
            # Calculate database size
            total_size = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
            click.echo(f"Database size: {total_size / 1024 / 1024:.2f} MB")
        click.echo("")

        # Get all project statistics
        all_stats = _run_async(database.get_project_stats(None))
        click.echo(f"📊 Total documents across all projects: {all_stats.get('total', 0)}")
        click.echo("")

        # Get table and show project breakdown
        table = database._ensure_connected()
        df = table.to_pandas()

        if len(df) == 0:
            click.echo("⚠️  Database is empty")
            return

        # Group by project
        project_counts = df.groupby("project_id").size().sort_values(ascending=False)

        click.echo("📁 Projects in database:")
        for pid, count in project_counts.items():
            marker = "👉" if pid == project_id else "  "
            click.echo(f"{marker} {pid}: {count} chunks")
        click.echo("")

        # Show document type statistics for specific project or all
        if project_id:
            project_df = df[df["project_id"] == project_id]
            if len(project_df) == 0:
                click.echo(f"⚠️  No documents found for project: {project_id}")
                return

            click.echo(f"📈 Document types for project {project_id}:")
            type_counts = project_df.groupby("doc_type").size()
            for doc_type, count in type_counts.items():
                click.echo(f"   {doc_type}: {count}")
            click.echo("")

            # Show sample documents
            click.echo(f"📄 Sample documents (limit: {limit}):")
            samples = project_df.head(limit)
            for _idx, row in samples.iterrows():
                click.echo(f"   - [{row['doc_type']}] {row['name']}")
                click.echo(f"     File: {row['file_path']}")
                if row["start_line"] > 0:
                    click.echo(f"     Lines: {row['start_line']}-{row['end_line']}")
                click.echo("")
        else:
            # Show overall document type breakdown
            click.echo("📈 Document type breakdown (all projects):")
            type_counts = df.groupby("doc_type").size()
            for doc_type, count in type_counts.items():
                click.echo(f"   {doc_type}: {count}")

    except Exception as e:
        click.echo(f"❌ Error inspecting database: {e!s}", err=True)
        import traceback

        click.echo(traceback.format_exc(), err=True)
