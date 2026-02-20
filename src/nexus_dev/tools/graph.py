"""Graph database tools for Nexus-Dev."""

from __future__ import annotations

import logging

from nexus_dev.app_state import get_config, get_hybrid_database, get_hybrid_db

logger = logging.getLogger(__name__)


async def search_dependencies(
    target: str,
    direction: str = "both",
    depth: int = 1,
    project_id: str | None = None,
) -> str:
    """Find code dependencies using the graph database.

    Use this tool to find what imports a file, what a file imports,
    or the full dependency tree. This gives EXACT results, not fuzzy matches.

    Args:
        target: File path or module name to search for.
        direction: 'imports' (what target imports),
                  'imported_by' (what imports target),
                  'both' (default).
        depth: How many levels deep to traverse (default: 1, max: 5).
        project_id: Optional project filter.

    Returns:
        List of dependent/dependency files with relationships.

    Examples:
        - search_dependencies("auth.py", direction="imported_by")
          → Files that import auth.py
        - search_dependencies("main.py", direction="imports", depth=2)
          → All imports of main.py and their imports
    """
    try:
        hybrid_db = get_hybrid_database()
    except RuntimeError as e:
        return f"Error: {e!s}"

    # Cap depth to prevent excessive queries
    depth = min(max(1, depth), 5)
    results = []

    try:
        if direction in ("imports", "both"):
            # What does target import?
            query = f"""
                MATCH import_path = (f:File {{path: $target}})-[:IMPORTS*1..{depth}]->(dep:File)
                RETURN dep.path AS dependency, length(import_path) AS distance
                ORDER BY distance
            """
            logger.debug(f"Executing imports query: {query} with target={target}")
            logger.debug(f"Graph name: {hybrid_db.graph.graph_name}")
            logger.debug(f"Graph client: {id(hybrid_db.graph.client)}")
            imports = hybrid_db.graph.query(query, {"target": target})
            logger.debug(f"Imports result: {imports.result_set}")

            if imports.result_set:
                results.append("## Imports (what this file depends on)")
                for row in imports.result_set:
                    indent = "  " * (row[1] - 1)  # distance is at index 1
                    results.append(f"{indent}→ {row[0]}")  # dependency is at index 0

        if direction in ("imported_by", "both"):
            # What imports target?
            query = f"""
                MATCH import_path = (f:File)-[:IMPORTS*1..{depth}]->(target:File {{path: $target}})
                RETURN f.path AS importer, length(import_path) AS distance
                ORDER BY distance
            """
            importers = hybrid_db.graph.query(query, {"target": target})

            if importers.result_set:
                if results:
                    results.append("")
                results.append("## Imported By (files that depend on this)")
                for row in importers.result_set:
                    indent = "  " * (row[1] - 1)  # distance is at index 1
                    results.append(f"{indent}← {row[0]}")  # importer is at index 0

        if not results:
            return (
                f"No dependencies found for '{target}'. "
                "Make sure the file is indexed with graph extraction enabled."
            )

        return "\n".join(results)

    except Exception as e:
        return f"Error querying dependencies: {e!s}"


async def find_callers(
    function_name: str,
    project_id: str | None = None,
) -> str:
    """Find all functions that call the specified function.

    Use this to understand the impact of changing a function.

    Args:
        function_name: Name of the function to find callers for.
        project_id: Optional project filter.

    Returns:
        List of calling functions with file locations.

    Example:
        find_callers("validate_user")
        → main.py:handle_login calls validate_user
        → api.py:auth_middleware calls validate_user
    """
    try:
        hybrid_db = get_hybrid_database()
    except RuntimeError as e:
        return f"Error: {e!s}"

    try:
        query = """
            MATCH (caller:Function)-[:CALLS]->(target:Function)
            WHERE target.name = $name
            RETURN caller.name AS caller_name,
                   caller.file_path AS file,
                   caller.start_line AS line,
                   target.name AS target_name
        """
        result = hybrid_db.graph.query(query, {"name": function_name})

        if not result.result_set:
            return (
                f"No callers found for function '{function_name}'. "
                "Make sure the code is indexed with graph extraction enabled."
            )

        lines = [f"## Functions that call `{function_name}`\n"]
        for row in result.result_set:
            caller_name = row[0]
            file_path = row[1]
            line_num = row[2]
            lines.append(f"- `{caller_name}` in {file_path}:{line_num}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error querying callers: {e!s}"


async def find_implementations(
    class_name: str,
    project_id: str | None = None,
) -> str:
    """Find all classes that inherit from the specified class.

    Use this to understand class hierarchies and find implementations.

    Args:
        class_name: Name of the base class.
        project_id: Optional project filter.

    Returns:
        Class hierarchy tree.

    Example:
        find_implementations("BaseHandler")
        → AuthHandler in handlers/auth.py
        → APIHandler in handlers/api.py
    """
    try:
        hybrid_db = get_hybrid_database()
    except RuntimeError as e:
        return f"Error: {e!s}"

    try:
        query = """
            MATCH path = (child:Class)-[:INHERITS*1..3]->(parent:Class)
            WHERE parent.name = $name
            RETURN child.name AS child_name,
                   child.file_path AS file,
                   length(path) AS depth
            ORDER BY depth
        """
        result = hybrid_db.graph.query(query, {"name": class_name})

        if not result.result_set:
            return (
                f"No subclasses found for '{class_name}'. "
                "Make sure the code is indexed with graph extraction enabled."
            )

        lines = [f"## Classes inheriting from `{class_name}`\n"]
        for row in result.result_set:
            child_name = row[0]
            file_path = row[1]
            depth = row[2]
            indent = "  " * (depth - 1)
            lines.append(f"{indent}└─ `{child_name}` in {file_path}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error querying implementations: {e!s}"


async def find_related_entities(
    query: str,
    session_id: str | None = None,
    limit: int = 10,
) -> str:
    """Find entities related to the query from session context.

    Use this to discover what code elements and concepts have been
    discussed together in this session.

    Args:
        query: Entity or concept to find relations for.
        session_id: Optional session filter.
        limit: Maximum results.

    Returns:
        Related entities with relationship types.
    """
    config = get_config()
    if not config or not config.enable_hybrid_db:
        return "Hybrid database not enabled. Set enable_hybrid_db=True in nexus_config.json."

    try:
        hybrid_db = get_hybrid_db()
        if hybrid_db is None:
            return "Failed to connect to hybrid database."

        # Default session if not provided
        effective_session = session_id or "default"

        # Find entity nodes matching query
        search_query = """
            MATCH (e:Entity)-[r:DISCUSSED|RELATED_TO*1..2]-(related)
            WHERE e.session_id = $session
              AND (e.name CONTAINS $query OR related.name CONTAINS $query)
            RETURN DISTINCT related.name AS name,
                   labels(related) AS labels,
                   related.entity_type AS entity_type
            LIMIT $limit
        """

        result = hybrid_db.graph.query(
            search_query,
            {"session": effective_session, "query": query, "limit": limit},
        )

        if not result.result_set:
            return f"No related entities found for '{query}' in session '{effective_session}'."

        lines = [f"## Entities related to '{query}'\n"]
        for row in result.result_set:
            name = row[0]
            labels = row[1] if row[1] else []
            entity_type = row[2] or (labels[0] if labels else "Entity")
            lines.append(f"- **{name}** ({entity_type})")

        return "\n".join(lines)

    except Exception as e:
        return f"Entity search failed: {e!s}"
