# Custom Agents

Deep dive into creating and customizing AI agents for your project.

---

## Overview

Custom agents are specialized AI personas that can be invoked via MCP tools. They combine:

- **Persona**: Role, goal, backstory, and tone
- **Memory**: RAG-enabled context from your knowledge base
- **Tools**: Access to Nexus-Dev functions
- **Model**: LLM configuration preferences

---

## Creating Agents

### From Template

```bash
# List available templates
nexus-agent templates

# Create from template
nexus-agent init code_reviewer --from-template code_reviewer
```

### Interactive

```bash
nexus-agent init my_agent
# Follow prompts
```

### Manual

Create `agents/<name>.yaml` directly:

```yaml
name: "project_expert"
display_name: "Project Expert"
description: "Deep knowledge of this specific codebase."

profile:
  role: "Senior Project Developer"
  goal: "Provide accurate, context-aware assistance for this codebase"
  backstory: |
    I have worked on this project since its inception. 
    I know every module, every quirk, and every design decision.
  tone: "Helpful and precise"

memory:
  enabled: true
  rag_limit: 10
  search_types: ["code", "documentation", "lesson", "insight"]

tools: []

llm_config:
  model_hint: "claude-sonnet-4.5"
  fallback_hints: ["auto"]
  temperature: 0.3
  max_tokens: 8000
```

---

## Configuration Deep Dive

### Profile Section

| Field | Purpose | Example |
|-------|---------|---------|
| `role` | Agent's expertise | "Security Expert" |
| `goal` | What agent optimizes for | "Find vulnerabilities" |
| `backstory` | Context for behavior | Multi-line description |
| `tone` | Response style | "Direct and technical" |

### Memory Section

```yaml
memory:
  enabled: true          # Use RAG for context
  rag_limit: 10          # Number of results to include
  search_types:          # What to search
    - code
    - documentation
    - lesson
    - insight
    - implementation
```

!!! tip "RAG Limit Tuning"
    Higher limits provide more context but increase token usage. Start with 5-10.

### Tools Section

```yaml
# All tools available (default)
tools: []

# Specific tools only
tools:
  - search_code
  - search_docs
  - record_lesson

# Exclude specific tools
tools:
  exclude:
    - invoke_tool
    - get_tool_schema
```

### LLM Configuration

```yaml
llm_config:
  model_hint: "claude-sonnet-4.5"   # Preferred model
  fallback_hints:                    # If preferred unavailable
    - "claude-haiku-3.5"
    - "auto"                         # Let IDE choose
  temperature: 0.5                   # 0=deterministic, 1=creative
  max_tokens: 4000                   # Response length limit
```

---

## Template Reference

### code_reviewer

Reviews code for quality, bugs, and best practices.

```yaml
profile:
  role: "Senior Code Reviewer"
  goal: "Identify bugs, security issues, and suggest improvements"
  tone: "Professional and constructive"
```

### doc_writer

Creates and maintains technical documentation.

```yaml
profile:
  role: "Technical Writer"
  goal: "Create clear, comprehensive documentation"
  tone: "Clear and instructional"
```

### debug_detective

Analyzes errors and proposes fixes.

```yaml
profile:
  role: "Debug Expert"
  goal: "Find root causes and propose targeted fixes"
  tone: "Analytical and methodical"
```

### security_auditor

Identifies vulnerabilities and security issues.

```yaml
profile:
  role: "Security Expert"
  goal: "Identify vulnerabilities and recommend remediations"
  tone: "Thorough and cautious"
```

### test_engineer

Generates test cases and improves coverage.

```yaml
profile:
  role: "Test Engineer"
  goal: "Ensure comprehensive test coverage"
  tone: "Precise and thorough"
```

---

## Advanced Patterns

### Project-Specific Agent

Create an agent with deep project knowledge:

```yaml
name: "project_oracle"
profile:
  role: "Project Oracle"
  goal: "Answer any question about this codebase accurately"
  backstory: |
    I have complete knowledge of:
    - The authentication system using JWT tokens
    - The database layer with PostgreSQL and Redis
    - The API design following REST principles
    - Historical bugs and their solutions
  tone: "Authoritative and helpful"

memory:
  enabled: true
  rag_limit: 15
  search_types: ["code", "documentation", "lesson", "insight", "implementation"]
```

### Domain Expert

For specific technical domains:

```yaml
name: "database_expert"
profile:
  role: "Database Architect"
  goal: "Optimize database design and queries"
  backstory: |
    Specialist in PostgreSQL, with expertise in:
    - Query optimization and EXPLAIN analysis
    - Index design and maintenance
    - Connection pooling and performance
  tone: "Technical and data-driven"
```

### Multi-Agent Workflow

Use multiple agents for complex tasks:

```
1. Ask debug_detective to analyze the error
2. Ask code_reviewer to review the proposed fix
3. Ask test_engineer to suggest test cases
```

---

## Best Practices

1. **Start with templates** - Customize rather than build from scratch
2. **Tune RAG limits** - Balance context vs token usage
3. **Be specific in backstory** - Include project-specific knowledge
4. **Test incrementally** - Verify agent behavior before heavy customization
5. **Version control agents** - Commit `agents/*.yaml` to git

---

## Troubleshooting

### Agent Not Appearing

```bash
# Refresh agents
refresh_agents()

# Or restart MCP server
```

### Poor Context

Increase RAG limit or ensure relevant content is indexed.

### Wrong Model Used

The IDE chooses the final model. `model_hint` is a preference, not a guarantee.

---

## See Also

- [nexus-agent CLI](../cli/agent.md) - Agent management commands
- [Agent Tools](../tools/agents.md) - Using agents via MCP
