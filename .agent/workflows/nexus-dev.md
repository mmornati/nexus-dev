---
description: Nexus-Dev Usage (Knowledge Base & RAG)
---

This project uses `nexus-dev`, a persistent memory and local RAG system (LanceDB) connected via MCP.

🎯 **STRICT CODE EXPLORATION RULE:**
You MUST NOT use your native global search tools or blindly read files to discover the codebase. You must consistently prioritize the semantic search provided by `nexus-dev`.

**Mandatory Procedure:**
1. ALWAYS use the `search_knowledge` MCP tool (or nexus-dev's sub-agents) FIRST to query the database about the architecture, functions, or project history.
2. Analyze the semantic context returned by the RAG.
3. ONLY THEN, if you need to see the full implementation of a specific file targeted by the RAG, are you allowed to use your native tools to open that specific file (using absolute paths).