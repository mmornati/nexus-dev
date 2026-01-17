# Model Reference Guide

*Updated: January 2026*

This guide helps you choose the right AI model for your Nexus-Dev agents based on your IDE and task requirements.

## Quick Reference

| Your IDE | Recommended Default | Best for Coding | Best for Writing |
|----------|---------------------|-----------------|------------------|
| **Antigravity** | `gemini-3-pro` | `gemini-3-deep-think` | `claude-opus-4.5` |
| **Cursor** | `claude-sonnet-4.5` | `gpt-5.2-codex` | `claude-opus-4.5` |
| **GitHub Copilot** | `gpt-5.2-codex` | `gpt-5.2-codex` | `claude-sonnet-4.5` |
| **VS Code (Copilot)** | `auto` | `gpt-5.2-codex` | `claude-sonnet-4.5` |
| **Continue** | Depends on BYOK | Provider-specific | Provider-specific |

---

## Model Families

### Claude (Anthropic)

| Model | Context | Pricing | Best For |
|-------|---------|---------|----------|
| **claude-opus-4.5** | 200K tokens | $15 / $75 per 1M tokens | Complex reasoning, technical writing, architecture |
| **claude-sonnet-4.5** | 200K tokens | $3 / $15 per 1M tokens | Balanced tasks, code review, everyday coding |
| **claude-haiku-4.5** | 200K tokens | $0.25 / $1.25 per 1M tokens | Fast responses, simple queries |

**Strengths**: Excellent code understanding, nuanced reasoning, strong writing quality

**Availability**: Antigravity, Cursor, Copilot, VS Code

### GPT (OpenAI)

| Model | Context | Pricing | Best For |
|-------|---------|---------|----------|
| **gpt-5.2** | 128K tokens | $5 / $15 per 1M tokens | General tasks, reasoning, multimodal |
| **gpt-5.2-codex** | 128K tokens | $10 / $30 per 1M tokens | Code generation, bug fixes, refactoring |
| **gpt-5-mini** | 128K tokens | $0.15 / $0.60 per 1M tokens | Fast tasks, low-cost operations |
| **o4-mini** | 128K tokens | $3 / $12 per 1M tokens | Advanced reasoning, slower but deeper |

**Strengths**: Fast code generation, broad knowledge, good at following instructions

**Availability**: Cursor, Copilot, VS Code

### Gemini (Google)

| Model | Context | Pricing | Best For |
|-------|---------|---------|----------|
| **gemini-3-pro** | 2M tokens | $1.25 / $5 per 1M tokens | Multimodal, large context, balanced tasks |
| **gemini-3-flash** | 1M tokens | $0.075 / $0.30 per 1M tokens | Fast responses, low latency |
| **gemini-3-deep-think** | 2M tokens | $10 / $30 per 1M tokens | Extended reasoning, complex analysis |

**Strengths**: Huge context windows, multimodal capabilities, native Antigravity support

**Availability**: Antigravity (native), Cursor, Copilot, VS Code (via extensions)

---

## IDE-Specific Configuration

### Antigravity

Antigravity uses native Gemini 3 models by default.

```yaml
llm_config:
  model_hint: "gemini-3-pro"
  fallback_hints: ["gemini-3-flash", "claude-sonnet-4.5", "auto"]
```

**Available Models**:
- `gemini-3-pro` (default)
- `gemini-3-flash`
- `gemini-3-deep-think`
- `claude-sonnet-4.5`
- `claude-opus-4.5`
- `gpt-oss-120b` (open-source variant)

### Cursor

Cursor supports all major model families with a model picker UI.

```yaml
llm_config:
  model_hint: "claude-sonnet-4.5"
  fallback_hints: ["gpt-5.2-codex", "gemini-3-pro", "auto"]
```

**Available Models**:
- `claude-sonnet-4.5` (recommended default)
- `claude-opus-4.5`
- `claude-haiku-4.5`
- `gpt-5.2`
- `gpt-5.2-codex`
- `gpt-5-mini`
- `gemini-3-pro`

### GitHub Copilot

Copilot uses auto-selection by default but allows manual model picking.

```yaml
llm_config:
  model_hint: "gpt-5.2-codex"
  fallback_hints: ["claude-sonnet-4.5", "gemini-2.5-pro", "auto"]
```

**Available Models**:
- `gpt-5.2-codex` (default for code)
- `gpt-5.1`
- `claude-sonnet-4.5`
- `claude-opus-4.5`
- `gemini-2.5-pro`
- `o4-mini`

### VS Code

VS Code requires extensions like Copilot, Continue, or AI Toolkit.

```yaml
llm_config:
  model_hint: "auto"  # Let the IDE choose
  fallback_hints: ["claude-sonnet-4.5", "gpt-5.2-codex"]
```

**Available Models**: Depends on installed extensions (Copilot, Continue, etc.)

---

## Fallback Strategy

Since Nexus-Dev uses **MCP Sampling**, the IDE ultimately selects the model. Our hints are preferences, not requirements.

### Recommended Fallback Chains

**For Code Analysis**:
```yaml
fallback_hints:
  - "claude-sonnet-4.5"  # Best reasoning
  - "gpt-5.2-codex"      # Code-optimized
  - "gemini-3-pro"       # Cross-platform
  - "auto"               # IDE chooses
```

**For Documentation**:
```yaml
fallback_hints:
  - "claude-opus-4.5"    # Best writing
  - "claude-sonnet-4.5"  # Still excellent
  - "gpt-5.2"            # Good alternative
  - "auto"               # IDE chooses
```

**For Debugging**:
```yaml
fallback_hints:
  - "claude-sonnet-4.5"    # Precise analysis
  - "gemini-3-deep-think"  # Deep reasoning
  - "gpt-5.2-codex"        # Code expertise
  - "auto"                 # IDE chooses
```

---

## Model Selection Tips

1. **Use `auto` for simplicity**: Let your IDE pick the best available model
2. **Claude for reasoning**: Best for complex code analysis and architecture
3. **GPT-Codex for generation**: Fastest code generation and refactoring
4. **Gemini for large context**: When you need to process entire files/modules
5. **Check your IDE**: Not all models are available in all IDEs

---

## Cost Optimization

| Strategy | Model Recommendation |
|----------|---------------------|
| **Budget-conscious** | `gpt-5-mini`, `claude-haiku-4.5`, `gemini-3-flash` |
| **Balanced** | `claude-sonnet-4.5`, `gpt-5.2-codex`, `gemini-3-pro` |
| **Maximum quality** | `claude-opus-4.5`, `o4-mini`, `gemini-3-deep-think` |

---

## Next Steps

- View available templates: `nexus-agent templates`
- Create agent from template: `nexus-agent init my_reviewer -t code_reviewer`
- Customize model: `nexus-agent init my_agent -t code_reviewer --model gemini-3-pro`
