# ADR-001: Session Context Cache for Ephemeral Development Memory

| Status   | Proposed |
|----------|----------|
| Date     | 2026-01-26 |
| Authors  | @mmornati |

## Context

### Problem Statement

AI coding assistants (LLMs) are stateless by design. They treat each request independently, leading to:

1. **Repetitive problem-solving** - The same solution is discovered multiple times in a session
2. **Cross-thread amnesia** - Discoveries in one chat thread don't transfer to another
3. **Context loss** - Long conversations lose earlier context as the window fills
4. **Delayed learning** - Valuable insights never become permanent lessons

### Industry Research

| Project | Type | Key Pattern |
|---------|------|------------|
| **[Mem0](https://github.com/mem0ai/mem0)** | Memory layer | Hybrid vector+graph+KV with self-improving memory |
| **[MemGPT](https://github.com/cpacker/MemGPT)** | Tiered memory | LLM self-manages main/archival memory |
| **[Zep](https://getzep.com)** | Session memory | Temporal knowledge graphs with entity relationships |
| **[LangChain](https://langchain.com)** | Framework | Buffer, summary, and conversation memory types |
| **Cursor Memory Bank** | File-based | `.cursor/memory/` with structured markdown |

These projects demonstrate that **embedding-based session memory improves LLM accuracy and reduces repetition**.

## Decision

**Add a session embedding cache to nexus-dev** with the following capabilities:

1. **New `SESSION_CONTEXT` document type** for ephemeral entries
2. **Session-scoped storage** with `session_id` and `expires_at` fields
3. **MCP tools** for caching, searching, and promoting session context
4. **TTL-based cleanup** with configurable expiration (default: 24h)
5. **Promotion workflow** to convert valuable discoveries to permanent lessons

## Technical Details

### Schema Changes

```python
class DocumentType(str, Enum):
    # ... existing types ...
    SESSION_CONTEXT = "session_context"

@dataclass
class Document:
    # ... existing fields ...
    session_id: str = ""              # Session identifier
    expires_at: datetime | None = None # Optional TTL expiration
    source_type: str = ""              # prompt | response | decision | discovery
```

### New MCP Tools

| Tool | Purpose |
|------|---------|
| `cache_session_context` | Store ephemeral context with optional TTL |
| `search_session_context` | Semantic search within current session |
| `promote_to_lesson` | Convert session entry to permanent lesson |
| `cleanup_session` | Explicitly clean up a session's cache |

### Session Identification Strategy

Sessions are identified by:
- **Conversation ID** when available from IDE/MCP context
- **Project ID + Date** for daily session buckets (fallback)
- **User-defined** session names via explicit parameter

## Expected Performance

### Latency

| Operation | Expected Latency | Notes |
|-----------|-----------------|-------|
| Cache entry | ~100-200ms | Single embedding + upsert |
| Search session | ~50-100ms | Vector search with session filter |
| Promote to lesson | ~100-150ms | Re-embed + upsert as lesson |
| Cleanup session | ~50-100ms | Bulk delete by session_id |

### Storage

| Metric | Estimate |
|--------|----------|
| Entries per session | 20-100 typical |
| Bytes per entry | ~2KB (text) + 6KB (vector) ≈ 8KB |
| Session size | 160KB - 800KB |
| Daily cleanup | Automatic via TTL |

### Embedding Costs (OpenAI)

| Scenario | Tokens | Cost |
|----------|--------|------|
| 50 context entries | ~25K tokens | ~$0.003 |
| 50 search queries | ~5K tokens | ~$0.0006 |
| Daily session | ~30K tokens | ~$0.004 |

## Usage Patterns

### Automatic Session Context Caching

```python
# LLM discovers a solution and caches it
await cache_session_context(
    content="TypeScript strict null check requires explicit undefined handling in route params",
    context_type="discovery",
    session_id="project-abc-2026-01-26"
)
```

### Search Before "Thinking"

```python
# LLM searches session before generating response
results = await search_session_context(
    query="TypeScript null check error in route",
    limit=3
)
# If found, apply existing solution instead of re-deriving
```

### Promotion to Permanent Lesson

```python
# User or LLM promotes valuable discovery
await promote_to_lesson(
    session_context_id="uuid-of-discovery",
    problem="TypeScript strict mode fails with route params",
    solution="Use optional chaining: params?.id ?? 'default'"
)
```

## AGENTS.md Integration

LLMs should be instructed via `AGENTS.md` (or equivalent) to:

1. **Search session before answering** related questions
2. **Cache discoveries** after non-trivial problem-solving
3. **Promote valuable findings** to permanent lessons

Example instruction block:

```markdown
## Session Memory Usage

Before providing solutions, search for similar past work:
1. `search_session_context(query="<problem summary>")`
2. If relevant result found, apply directly
3. After solving new problems, cache with `cache_session_context(...)`
```

## Consequences

### Positive

- **Reduced repetition** - Don't re-solve identical problems
- **Cross-thread continuity** - Share context across chat threads
- **Gradual learning** - Organic path from ephemeral → permanent
- **Lower cost** - Cache hits avoid LLM re-computation

### Negative

- **Storage overhead** - Session data uses vector DB space
- **Complexity** - Another layer to understand and debug
- **Embedding cost** - Every cache operation has embedding cost
- **False positives** - Semantic search may return irrelevant matches

### Mitigations

| Risk | Mitigation |
|------|------------|
| Storage bloat | TTL expiration + storage limits |
| Embedding cost | Batch embedding, configurable TTL |
| False positives | Similarity threshold, human review for promotions |
| Complexity | Opt-in feature, sensible defaults |

## Alternatives Considered

1. **File-based only (like Cursor Memory Bank)** - Simpler but no semantic search
2. **External service (Mem0/Zep)** - More features but adds dependency
3. **No session cache** - Rely only on permanent lessons (current state)

**Decision**: Build native session cache. Leverages existing LanceDB infrastructure, no external dependencies, and provides semantic search.

## References

- [Mem0 GitHub](https://github.com/mem0ai/mem0)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Zep - Long-Term Memory for LLM Agents](https://getzep.com)
- [LangChain Memory Concepts](https://python.langchain.com/docs/concepts/memory/)
