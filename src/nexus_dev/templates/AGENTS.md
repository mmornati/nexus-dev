# {project_name} - AI Agent Instructions

## Project Overview

<!-- Provide a brief description of the project here -->

## Nexus-Dev Knowledge Base

This project uses nexus-dev for persistent AI memory.

**Project ID:** {project_id}

### Mandatory Search-First Workflow

BEFORE answering questions about this code:
1. `search_knowledge("<query>")` - Search code, docs, and lessons
2. `search_lessons("<error>")` - Check for past solutions
3. `search_session_context("<query>")` - Check this session's discoveries

### Recording Knowledge

After solving non-trivial problems:
- `record_lesson(problem="...", solution="...")` - Permanent lesson
- `cache_session_context(content="...", context_type="discovery")` - Session cache
- `record_insight(category="mistake|discovery", ...)` - Capture reasoning

### Session Best Practice

- Start each session: `get_project_context()`
- Search before implementing: `search_session_context("<intent>")`
- Cache discoveries: `cache_session_context(...)`
- Promote valuable findings: `promote_to_lesson(...)`

## Additional Instructions

<!-- User can add custom content here -->
