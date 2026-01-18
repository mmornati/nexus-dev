# Nexus-Dev

**Persistent Memory for AI Coding Agents**

Nexus-Dev is an open-source MCP (Model Context Protocol) server that provides a local RAG (Retrieval-Augmented Generation) system for AI coding assistants like GitHub Copilot, Cursor, and Windsurf. It learns from your codebase and mistakes, enabling cross-project knowledge sharing.

---

## ✨ Key Features

<div class="grid cards" markdown>

-   :brain: **Persistent Memory**

    ---

    Index your code and documentation for semantic search. Your AI assistant remembers context across sessions.

-   :books: **Lesson Learning**

    ---

    Record problems and solutions. The AI recalls them when similar issues arise.

-   :octicons-mark-github-16: **GitHub Integration**

    ---

    Import Issues and Pull Requests into your knowledge base for context-aware assistance.

-   :globe_with_meridians: **Multi-Language Support**

    ---

    Python, JavaScript/TypeScript, Java - with extensibility via tree-sitter.

-   :arrows_counterclockwise: **Cross-Project Learning**

    ---

    Share knowledge across all your projects. Solutions from one project help others.

-   :house: **Local-First**

    ---

    All data stays on your machine with LanceDB. Privacy by default.

</div>

---

## 🚀 Quick Start

Install Nexus-Dev globally (no project clone needed):

=== "pipx (Recommended)"

    ```bash
    pipx install nexus-dev
    ```

=== "uv"

    ```bash
    uv tool install nexus-dev
    ```

Then initialize your project:

```bash
cd your-project
nexus-init --project-name "my-project" --embedding-provider openai
nexus-index src/ -r
```

Configure your IDE and you're ready! See the [Quick Start Guide](quickstart.md) for full details.

---

## 📚 Documentation

| Section | Description |
|---------|-------------|
| [Quick Start](quickstart.md) | 5-minute setup guide |
| [Installation](getting-started/installation.md) | Detailed installation options |
| [CLI Reference](cli/index.md) | All command-line tools |
| [MCP Tools](tools/index.md) | Tools exposed to AI agents |
| [Workflows](workflows/new-project.md) | Step-by-step usage guides |

---

## 🔧 MCP Tools Overview

Nexus-Dev exposes powerful tools to your AI coding assistant:

### Search Tools

| Tool | Description |
|------|-------------|
| `search_knowledge` | Search all content (code, docs, lessons) |
| `search_code` | Find function and class definitions |
| `search_docs` | Search documentation |
| `search_lessons` | Find past solutions to problems |

### Learning Tools

| Tool | Description |
|------|-------------|
| `record_lesson` | Store a problem/solution pair |
| `record_insight` | Capture discoveries and mistakes |
| `record_implementation` | Document completed features |

### Gateway Tools

| Tool | Description |
|------|-------------|
| `search_tools` | Find MCP tools across all servers |
| `invoke_tool` | Execute tools on backend servers |

---

## 🏗️ Architecture

```mermaid
flowchart TB
    subgraph Agent["🤖 AI Agent"]
        direction TB
        IDE["Cursor / Copilot / Windsurf"]
    end

    subgraph MCP["📡 Nexus-Dev"]
        direction TB
        Tools["MCP Tools"]
        RAG["RAG Pipeline"]
        DB["LanceDB"]
    end

    subgraph External["🌍 External MCP Servers"]
        GH["GitHub"]
        HA["Home Assistant"]
        Other["..."]
    end

    Agent -->|"stdio"| Tools
    Tools --> RAG
    RAG --> DB
    Tools -.->|"gateway"| External
```

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](https://github.com/mmornati/nexus-dev/blob/main/CONTRIBUTING.md) for guidelines.

---

## 📄 License

MIT License - see [LICENSE](https://github.com/mmornati/nexus-dev/blob/main/LICENSE) for details.
