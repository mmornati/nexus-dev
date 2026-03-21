# Nexus-Dev Implementation User Stories

This document contains detailed user stories for implementing the Nexus-Dev improvements planned for Phase 1-4.

---

## Overview

**Project Goal:** Improve Nexus-Dev's MCP Gateway to make it easier for LLMs to use correctly, while reducing token usage through caching and summarization.

**Current State:**
- Nexus-Dev provides 3-step gateway workflow: `search_tools` → `get_tool_schema` → `invoke_tool`
- LLMs often don't understand this pattern and try to use tools directly
- No caching or summarization of tool results

**Target State:**
- LLMs clearly understand the gateway workflow
- Tool results are cached to reduce repeated API calls
- Tool outputs are summarized to reduce token usage

---

## User Story Format

Each user story follows this format:

```
## US-XXX: [Title]

**Priority:** P0 | P1 | P2  
**Phase:** 1 | 2 | 3 | 4  
**Estimate:** XS | S | M | L | XL

### User Story
As a [role], I want [feature] so that [benefit].

### Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### Technical Implementation Notes
- File: `src/nexus_dev/xxx.py`
- Function: `xxx_function()`
- Dependencies: None | requires US-XXX

### Example
```
// Input / Usage example
```

### Related User Stories
- Related US-XXX
```

---

## Phase 1: Gateway Workflow Clarity (Priority: P0)

### US-101: Enhanced search_tools Description

**Priority:** P0  
**Phase:** 1  
**Estimate:** S

#### User Story
As an LLM using Nexus-Dev, I want the `search_tools` function to clearly explain the gateway workflow so that I understand I must use this tool FIRST before accessing external MCP tools.

#### Acceptance Criteria
- [ ] `search_tools` tool description contains explicit 3-step workflow
- [ ] Description includes example showing the full workflow
- [ ] Description warns against calling external tools directly
- [ ] Description is in plain English understandable by LLMs

#### Technical Implementation Notes
- File: `src/nexus_dev/tools/mcp_tools.py`
- Function: `search_tools()`
- The description parameter in `@mcp.tool()` decorator should be updated

#### Example
```
Tool: search_tools(query: str, server: str | None = None, limit: int = 5) -> str

Description:
"Find the RIGHT tool from other MCP servers to complete a task.

This is your FIRST STEP when you need to do something outside Nexus-Dev:
1. Use search_tools('<action>') to find available tools
2. Use get_tool_schema(server, tool) to see parameters
3. Use invoke_tool(server, tool, arguments) to execute

Example: User asks to 'create a GitHub issue'
→ search_tools('create issue on GitHub')
→ Returns: github.create_issue with schema
→ get_tool_schema('github', 'create_issue')
→ invoke_tool(server='github', tool='create_issue', arguments={...})

CRITICAL: Never try to call external tools directly (e.g., github.create_issue)!
The tool will not exist and will fail. Always use this gateway workflow."
```

#### Related User Stories
- US-102, US-103

---

### US-102: Enhanced invoke_tool Description

**Priority:** P0  
**Phase:** 1  
**Estimate:** S

#### User Story
As an LLM using Nexus-Dev, I want the `invoke_tool` function to clearly show it as the FINAL step of the gateway workflow, with examples of correct usage.

#### Acceptance Criteria
- [ ] `invoke_tool` description clarifies it's step 3 of 3
- [ ] Includes reference to step 1 (search_tools) and step 2 (get_tool_schema)
- [ ] Shows correct parameter format (server as separate string, tool as separate string)
- [ ] Warns against common mistakes (passing full tool name as single string)

#### Technical Implementation Notes
- File: `src/nexus_dev/tools/mcp_tools.py`
- Function: `invoke_tool()`

#### Example
```
Tool: invoke_tool(server: str, tool: str, arguments: dict | None = None) -> str

Description:
"Execute a tool on a backend MCP server through the Nexus-Dev gateway.

This is STEP 3 of the gateway workflow:
1. First: search_tools('<action>') to find the right tool
2. Then: get_tool_schema(server, tool) to see required parameters
3. Finally: invoke_tool(server, tool, arguments) to execute

Example workflow:
1. search_tools('create GitHub issue') → Returns: github.create_issue
2. get_tool_schema('github', 'create_issue') → Returns parameters
3. invoke_tool(
     server='github',           # NOT 'github.create_issue'!
     tool='create_issue',      # Tool name WITHOUT server prefix
     arguments={
       'owner': 'myorg',
       'repo': 'myrepo',
       'title': 'Bug fix',
       'body': 'Fixed the issue'
     }
   )

CRITICAL: Pass server and tool as SEPARATE parameters!
Do NOT combine them like invoke_tool('github.create_issue', {...})"
```

#### Related User Stories
- US-101, US-103

---

### US-103: Update AGENTS.md Gateway Section

**Priority:** P0  
**Phase:** 1  
**Estimate:** S

#### User Story
As a developer configuring their AI agent, I want the AGENTS.md template to include a clear gateway usage section so that the LLM always follows the correct workflow.

#### Acceptance Criteria
- [ ] AGENTS.md template has "CRITICAL: Gateway Tool Usage" section
- [ ] Section shows correct vs incorrect usage examples
- [ ] Section explains why the gateway exists (reduce tool count)
- [ ] Section is placed prominently at the top of tools section

#### Technical Implementation Notes
- File: `docs/AGENTS_TEMPLATE.md`
- Add new section after "⚠️ CRITICAL: RAG Usage Policy"

#### Example
```markdown
## ⚠️ CRITICAL: Gateway Tool Usage

> **MANDATORY**: When you need to use tools from other MCP servers (GitHub, Home Assistant, Filesystem, etc), you MUST use the Nexus-Dev gateway.

### Why This Matters

Nexus-Dev acts as a gateway to reduce the number of tools you see. Instead of 50+ tools, you have ~20 core tools. To access external tools:

### Correct Workflow (3 Steps)

1. **Search** for the right tool:
   ```python
   search_tools("create a GitHub issue")
   ```

2. **Get** the tool schema (parameters):
   ```python
   get_tool_schema(server="github", tool="create_issue")
   ```

3. **Invoke** the tool:
   ```python
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
   ```

### Common Mistakes (DO NOT DO)

❌ WRONG - Trying to use external tools directly:
```python
# These tools don't exist in Nexus-Dev!
github.create_issue(...)    # ❌ Will fail!
homeassistant.turn_on(...)  # ❌ Will fail!
filesystem.read_file(...)   # ❌ Will fail!
```

❌ WRONG - Passing tool name with server prefix:
```python
invoke_tool("github.create_issue", {...})  # ❌ Wrong!
invoke_tool(server="github.create_issue", ...)  # ❌ Wrong!
```

✅ CORRECT - Server and tool as SEPARATE strings:
```python
invoke_tool(server="github", tool="create_issue", {...})  # ✅ Correct!
```

### Quick Reference

| Tool | Purpose |
|------|---------|
| search_tools | Find available tools from external servers |
| get_tool_schema | See what parameters a tool needs |
| invoke_tool | Execute a tool on an external server |
```

#### Related User Stories
- US-101, US-102

---

### US-104: Get Gateway System Prompt Tool

**Priority:** P1  
**Phase:** 1  
**Estimate:** M

#### User Story
As an LLM, I want a tool that returns gateway usage instructions that can be included in my system prompt so that I always have the workflow reference available.

#### Acceptance Criteria
- [ ] New tool `get_gateway_prompt()` exists
- [ ] Returns formatted markdown with gateway workflow
- [ ] Can be called to inject guidance into context
- [ ] Returns the same content as US-103 "Quick Reference" section

#### Technical Implementation Notes
- File: `src/nexus_dev/tools/system_prompts.py` (NEW)
- Function: `get_gateway_prompt()`
- Register tool in `server.py`

#### Example
```
Tool: get_gateway_prompt() -> str

Returns:
"## Gateway Tool Usage (MANDATORY)

When you need external MCP tools:
1. search_tools('<action>') - Find available tools
2. get_tool_schema(server, tool) - Get parameters  
3. invoke_tool(server, tool, args) - Execute

Example:
1. search_tools('create GitHub issue')
2. get_tool_schema('github', 'create_issue')
3. invoke_tool(server='github', tool='create_issue', arguments={...})

NEVER call external tools directly!"
```

#### Related User Stories
- US-101, US-102, US-103

---

## Phase 2: Token Optimization (Priority: P0)

### US-201: Tool Result Caching

**Priority:** P0  
**Phase:** 2  
**Estimate:** L

#### User Story
As a developer using Nexus-Dev, I want tool invocation results to be cached so that repeated calls to the same tool with the same arguments don't require additional API calls, reducing costs and latency.

#### Acceptance Criteria
- [ ] Cache stores results keyed by (server, tool, arguments_hash)
- [ ] Cache has configurable TTL (default: 5 minutes)
- [ ] Cache can be disabled per-tool via config (cache: false)
- [ ] Cache has maximum size limit with LRU eviction (default: 1000 entries)
- [ ] Mutations (create, update, delete) are NOT cached by default
- [ ] Cache is cleared on server profile change or refresh
- [ ] Cache hits return immediately without calling external server
- [ ] Configuration options in mcp_config.json:
  ```json
  {
    "gateway": {
      "cache": {
        "enabled": true,
        "ttl_seconds": 300,
        "max_entries": 1000
      }
    },
    "servers": {
      "github": {
        "cache": false
      }
    }
  }
  ```

#### Technical Implementation Notes
- File: `src/nexus_dev/gateway/cache.py` (NEW)
- Create `ToolCache` class with:
  - `get(key) -> result | None`
  - `set(key, result, ttl) -> None`
  - `invalidate(key) -> None`
  - `clear() -> None`
- Integrate into `connection_manager.py` `invoke_tool()` method
- Cache key: SHA256(f"{server}:{tool}:{sorted_json(arguments)}")

#### Cache Flow Diagram
```
invoke_tool(server, tool, arguments)
        │
        ▼
┌───────────────────┐
│ Generate cache key│
│ sha256(s:t:args)  │
└───────────────────┘
        │
        ▼
┌───────────────────┐     YES    ┌─────────────┐
│ Cache hit?        │──────────▶│ Return cached│
└───────────────────┘           │ result      │
        │ NO                    └─────────────┘
        ▼
┌───────────────────┐
│ Call actual tool  │
│ via connection    │
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Store in cache    │
│ (if not mutation) │
└───────────────────┘
        │
        ▼
    Return result
```

#### Related User Stories
- US-202, US-203

---

### US-202: Tool Output Summarization

**Priority:** P0  
**Phase:** 2  
**Estimate:** M

#### User Story
As a developer using Nexus-Dev, I want tool outputs to be summarized when they're too verbose so that I reduce token usage while still getting the essential information.

#### Acceptance Criteria
- [ ] Tool outputs can be truncated/summarized based on config
- [ ] Configurable max list items (default: 10)
- [ ] Configurable max output characters (default: 500)
- [ ] Summarization preserves: status, IDs, URLs, key metrics
- [ ] Configurable per-server in mcp_config.json:
  ```json
  {
    "servers": {
      "github": {
        "summarize": true,
        "max_list_items": 10,
        "max_output_chars": 500
      }
    }
  }
  ```
- [ ] For arrays: shows first N items + "X more items" count
- [ ] For objects: shows first N key-value pairs + "X more fields"
- [ ] Default summarization: enabled for all servers

#### Technical Implementation Notes
- File: `src/nexus_dev/gateway/summarizer.py` (NEW)
- Create `OutputSummarizer` class with:
  - `summarize(result, config) -> summarized_result`
  - `truncate_list(items, max_items) -> truncated`
  - `truncate_object(obj, max_fields) -> truncated`
- Integrate into `mcp_tools.py` `invoke_tool()` after getting result
- Handle different result types: string, JSON, array, object

#### Summarization Rules

**For Arrays/Lists:**
```
Input: ["repo1", "repo2", "repo3", "repo4", "repo5", "repo6", "repo7"]
Config: max_list_items = 3

Output:
- repo1
- repo2
- repo3
... (showing 3 of 7 items)
```

**For JSON Objects:**
```
Input: {"name": "repo", "stars": 100, "forks": 20, "watchers": 5, "language": "Python", ...}
Config: max_fields = 3

Output:
{
  "name": "repo",
  "stars": 100,
  "forks": 20,
  ... (showing 3 of 15 fields)
}
```

**For Long Strings:**
```
Input: "This is a very long output string that is over 500 characters..."
Config: max_output_chars = 100

Output: "This is a very long output string that is over 500 characters... [truncated, 1500 -> 100 chars]"
```

#### Related User Stories
- US-201, US-203

---

### US-203: Caching Configuration in MCP Config

**Priority:** P0  
**Phase:** 2  
**Estimate:** S

#### User Story
As a developer, I want to configure caching and summarization per-server in the mcp_config.json file so that I can optimize token usage for each external service.

#### Acceptance Criteria
- [ ] mcp_config.json schema supports caching options
- [ ] mcp_config.json schema supports summarization options
- [ ] Server-specific settings override global gateway settings
- [ ] Schema validation ensures valid configuration

#### Technical Implementation Notes
- File: `src/nexus_dev/mcp_config.py`
- Add `CacheSettings` dataclass:
  ```python
  @dataclass
  class CacheSettings:
      enabled: bool = True
      ttl_seconds: int = 300  # 5 minutes
      max_entries: int = 1000
  ```
- Add `SummarizeSettings` dataclass:
  ```python
  @dataclass
  class SummarizeSettings:
      enabled: bool = True
      max_list_items: int = 10
      max_output_chars: int = 500
  ```
- Add to `MCPServerConfig`:
  ```python
  cache: CacheSettings | None = None
  summarize: SummarizeSettings | None = None
  ```
- Update schema: `src/nexus_dev/schemas/mcp_config_schema.json`

#### Example Configuration
```json
{
  "version": "1.0",
  "gateway": {
    "default_timeout": 30,
    "cache": {
      "enabled": true,
      "ttl_seconds": 300,
      "max_entries": 1000
    },
    "summarize": {
      "enabled": true,
      "max_list_items": 10,
      "max_output_chars": 500
    }
  },
  "servers": {
    "github": {
      "transport": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "enabled": true,
      "cache": {
        "enabled": true,
        "ttl_seconds": 600
      },
      "summarize": {
        "enabled": true,
        "max_list_items": 5
      }
    },
    "homeassistant": {
      "transport": "sse",
      "url": "http://localhost:8123/mcp",
      "enabled": true,
      "cache": {
        "enabled": false
      }
    }
  }
}
```

#### Related User Stories
- US-201, US-202

---

## Phase 3: Better Automation (Priority: P1)

### US-301: Enhanced Git Commit Auto-Index

**Priority:** P1  
**Phase:** 3  
**Estimate:** M

#### User Story
As a developer, I want Nexus-Dev to automatically index changed files on git commit so that my knowledge base stays up-to-date without manual intervention.

#### Acceptance Criteria
- [ ] Pre-commit hook indexes changed files
- [ ] Hook shows summary: "Indexed X files → Y chunks"
- [ ] Hook is fast (< 5 seconds for typical commit)
- [ ] Hook handles errors gracefully (doesn't block commit)
- [ ] Hook can be skipped with `--no-verify` if needed

#### Technical Implementation Notes
- File: `src/nexus_dev/cli/index.py`
- Enhance existing `index` command to accept file paths
- Pre-commit hook in `src/nexus_dev/cli/init.py`

#### Example
```bash
$ git commit -m "Add authentication"
[nexus-dev] Indexing changed files...
[nexus-dev] Indexed 3 files → 42 chunks
[nexus-dev] Indexed in 2.3s
```

#### Related User Stories
- US-302

---

### US-302: MCP Tool Auto-Discovery on Startup

**Priority:** P1  
**Phase:** 3  
**Estimate:** M

#### User Story
As a developer, I want Nexus-Dev to automatically discover and index MCP tools on server startup so that I don't need to manually run `nexus-index-mcp`.

#### Acceptance Criteria
- [ ] Server checks for MCP config on startup
- [ ] If MCP config exists, automatically indexes tools
- [ ] Indexing happens in background (non-blocking)
- [ ] Logs show indexing progress
- [ ] Can be disabled via flag if needed

#### Technical Implementation Notes
- File: `src/nexus_dev/server.py`
- Add `--no-auto-index` CLI flag
- Add `_auto_index_mcp_tools()` function
- Call from `main()` after config load

#### Example
```
Starting Nexus-Dev MCP server...
Found MCP config: .nexus/mcp_config.json
Auto-indexing MCP tools...
Indexed 45 tools from 3 servers (github, homeassistant, filesystem)
Server initialization complete, running stdio transport
```

#### Related User Stories
- US-301, US-303

---

### US-303: Session-Aware Search Suggestions

**Priority:** P2  
**Phase:** 3  
**Estimate:** L

#### User Story
As an LLM, I want Nexus-Dev to remember what the user is working on and proactively suggest relevant searches so that I can provide better context.

#### Acceptance Criteria
- [ ] Session context is stored (current task, recent files)
- [ ] `get_project_context()` includes recent task context
- [ ] Suggestions are based on file being edited
- [ ] Suggestions can be triggered explicitly

#### Technical Implementation Notes
- File: `src/nexus_dev/kv_store.py` (extend)
- Store session context in KV store
- Add to `tools/context.py`

#### Related User Stories
- US-101, US-102

---

## Phase 4: Observability (Priority: P2)

### US-401: Gateway Usage Metrics

**Priority:** P2  
**Phase:** 4  
**Estimate:** M

#### User Story
As a developer, I want to see how the gateway is being used so that I can understand if LLMs are following the correct workflow.

#### Acceptance Criteria
- [ ] Log gateway tool usage (search_tools, invoke_tool)
- [ ] Track cache hit/miss ratio
- [ ] Track tools accessed per server
- [ ] Metrics accessible via CLI command

#### Technical Implementation Notes
- File: `src/nexus_dev/gateway/metrics.py` (NEW)
- Add `GatewayMetrics` class
- Add CLI command: `nexus gateway stats`

#### Example
```
$ nexus gateway stats
Gateway Usage (last 24h):
- search_tools calls: 45
- invoke_tool calls: 123
- Cache hits: 89 (72%)
- Cache misses: 34

Tools by server:
- github: 67 calls
- homeassistant: 34 calls  
- filesystem: 22 calls
```

#### Related User Stories
- US-201, US-202

---

### US-402: Debug Mode for Gateway

**Priority:** P2  
**Phase:** 4  
**Estimate:** S

#### User Story
As a developer debugging issues, I want verbose logging of gateway operations so that I can understand why certain tool calls fail or behave unexpectedly.

#### Acceptance Criteria
- [ ] Debug flag enables verbose logging
- [ ] Logs show cache hits/misses
- [ ] Logs show tool routing decisions
- [ ] Logs are readable and informative

#### Technical Implementation Notes
- Add `--debug` flag to server CLI
- Enhance logging in `connection_manager.py`

#### Related User Stories
- US-401

---

## Implementation Order

| Order | US | Title | Priority | Phase |
|-------|-----|-------|----------|-------|
| 1 | US-101 | Enhanced search_tools Description | P0 | 1 |
| 2 | US-102 | Enhanced invoke_tool Description | P0 | 1 |
| 3 | US-103 | Update AGENTS.md Gateway Section | P0 | 1 |
| 4 | US-203 | Caching Configuration in MCP Config | P0 | 2 |
| 5 | US-201 | Tool Result Caching | P0 | 2 |
| 6 | US-202 | Tool Output Summarization | P0 | 2 |
| 7 | US-104 | Get Gateway System Prompt Tool | P1 | 1 |
| 8 | US-301 | Enhanced Git Commit Auto-Index | P1 | 3 |
| 9 | US-302 | MCP Tool Auto-Discovery on Startup | P1 | 3 |
| 10 | US-401 | Gateway Usage Metrics | P2 | 4 |
| 11 | US-402 | Debug Mode for Gateway | P2 | 4 |
| 12 | US-303 | Session-Aware Search Suggestions | P2 | 3 |

---

## Technical Dependencies

```
US-101 ─────┐
            ├──► US-103
US-102 ─────┘        │
                     │
US-203 ──────────────┼──► US-201 ───► US-401
                     │          │
US-202 ──────────────┘          │
                     │
US-301 ──────────────┼──► US-302
                     │
US-104 ──────────────┘
```

---

## Notes for Implementation

1. **Start with Phase 1**: The description improvements will have immediate impact with minimal code changes.

2. **Caching (US-201) is foundational**: Many other features depend on it. Implement early.

3. **Configuration first**: Implement US-203 before US-201 and US-202 so the config structure is ready.

4. **Test incrementally**: Each user story should have tests before moving to the next.

5. **Backward compatibility**: Ensure existing functionality still works - these are enhancements, not breaking changes.
