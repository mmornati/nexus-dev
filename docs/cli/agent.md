# nexus-agent

Create and manage custom AI agents.

---

## Synopsis

```bash
nexus-agent COMMAND [OPTIONS]
```

---

## Description

Create specialized AI personas that can be invoked via MCP tools. Agents are defined in YAML files in the `agents/` directory.

---

## Commands

| Command | Description |
|---------|-------------|
| `init` | Create a new agent |
| `list` | List configured agents |
| `templates` | Show available templates |

---

## nexus-agent init

Create a new custom agent.

### From Template (Recommended)

```bash
nexus-agent init code_reviewer --from-template code_reviewer
```

### Interactive

```bash
nexus-agent init my_agent
# Follow prompts for role, goal, backstory
```

### With Custom Model

```bash
nexus-agent init security_expert \
  --from-template security_auditor \
  --model claude-opus-4.5
```

---

## nexus-agent templates

List available agent templates.

```bash
nexus-agent templates
```

**Output:**

```
Available Agent Templates:

  code_reviewer     - Reviews code for bugs, security issues, and best practices
  doc_writer        - Creates and updates technical documentation
  debug_detective   - Analyzes errors and proposes fixes
  refactor_architect - Suggests code restructuring and design patterns
  test_engineer     - Generates test cases and improves coverage
  security_auditor  - Identifies vulnerabilities and recommends fixes
  api_designer      - Reviews and designs REST/GraphQL APIs
  performance_optimizer - Finds performance bottlenecks
```

---

## nexus-agent list

Show configured agents.

```bash
nexus-agent list
```

**Output:**

```
Configured Agents:

  ✅ code_reviewer
     Role: Senior Code Reviewer
     Model: claude-sonnet-4.5

  ✅ my_custom_agent
     Role: Project Expert
     Model: auto
```

---

## Agent Configuration

Agents are defined in `agents/<name>.yaml`:

```yaml
name: "code_reviewer"
display_name: "Code Reviewer"
description: "Delegate code review tasks to the Code Reviewer agent."

profile:
  role: "Senior Code Reviewer"
  goal: "Identify bugs, security issues, and suggest improvements"
  backstory: "Expert developer with 10+ years of experience in code quality."
  tone: "Professional and constructive"

memory:
  enabled: true
  rag_limit: 5
  search_types: ["code", "documentation", "lesson"]

tools: []  # Empty = all tools available

llm_config:
  model_hint: "claude-sonnet-4.5"
  fallback_hints: ["auto"]
  temperature: 0.5
  max_tokens: 4000
```

---

## Using Agents

After creating an agent, use it via MCP:

```
Use the ask_code_reviewer tool to review this function for security issues.
```

Or invoke directly:

```bash
# Via MCP tool
ask_code_reviewer("Review the authentication module for security issues")
```

---

## See Also

- [Agent Tools](../tools/agents.md) - Using agents via MCP
- [Workflows](../workflows/new-project.md#optional-add-custom-agents) - Agent setup walkthrough
