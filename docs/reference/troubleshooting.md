# Troubleshooting

Common issues and their solutions.

---

## Installation Issues

### Command Not Found: nexus-init

**Symptom:**
```
zsh: command not found: nexus-init
```

**Solutions:**

1. **Check installation:**
   ```bash
   pip show nexus-dev
   ```

2. **Reinstall with pipx:**
   ```bash
   pipx uninstall nexus-dev
   pipx install nexus-dev
   pipx ensurepath
   ```

3. **Restart shell:**
   ```bash
   source ~/.zshrc  # or ~/.bashrc
   ```

### Python Version Error

**Symptom:**
```
ERROR: Package requires Python >=3.13
```

**Solution:**

Install Python 3.13+:
```bash
# macOS with pyenv
pyenv install 3.13
pyenv global 3.13

# Then reinstall
pipx reinstall nexus-dev
```

---

## Configuration Issues

### "nexus_config.json not found"

**Symptom:**
```
❌ nexus_config.json not found. Run 'nexus-init' first.
```

**Solutions:**

1. **Initialize the project:**
   ```bash
   nexus-init --project-name "My Project"
   ```

2. **Check working directory:**
   ```bash
   pwd  # Should be project root
   ls nexus_config.json
   ```

### OpenAI API Key Error

**Symptom:**
```
Error: OPENAI_API_KEY not set
```

**Solution:**

```bash
export OPENAI_API_KEY="sk-..."

# Or add to shell profile
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.zshrc
source ~/.zshrc
```

---

## Indexing Issues

### No Files Indexed

**Symptom:**
```
No files to index.
```

**Causes & Solutions:**

1. **Wrong patterns in config:**
   Check `nexus_config.json`:
   ```json
   {
     "include_patterns": ["**/*.py", "**/*.js"],
     "exclude_patterns": ["**/node_modules/**"]
   }
   ```

2. **Missing `-r` flag:**
   ```bash
   nexus-index src/ -r  # Recursive
   ```

3. **All files excluded:**
   Check if files match exclude patterns.

### Chunking Errors

**Symptom:**
```
❌ Failed to index file.py: syntax error
```

**Cause:** Invalid syntax in source file.

**Solution:** Fix syntax errors before indexing. The file is skipped, not fatal.

---

## MCP Server Issues

### Server Not Connecting

**Symptom:** AI agent doesn't see Nexus-Dev tools.

**Solutions:**

1. **Restart IDE** after configuration changes.

2. **Check MCP config syntax:**
   ```json
   {
     "mcpServers": {
       "nexus-dev": {
         "command": "nexus-dev",
         "args": []
       }
     }
   }
   ```

3. **Test server manually:**
   ```bash
   nexus-dev
   # Should start without errors
   # Ctrl+C to exit
   ```

### "Empty" Server (No Project Context)

**Symptom:** Server connects but has no project data.

**Solutions:**

1. **Set project root:**
   ```json
   {
     "env": {
       "NEXUS_PROJECT_ROOT": "/path/to/project"
     }
   }
   ```

2. **Use refresh_agents:**
   ```
   Call the refresh_agents tool
   ```

---

## Search Issues

### No Results Found

**Symptom:**
```
No results found for query: 'function name'
```

**Solutions:**

1. **Check if indexed:**
   ```bash
   nexus-status
   ```
   Should show non-zero chunk counts.

2. **Re-index:**
   ```bash
   nexus-reindex
   ```

3. **Try broader query:**
   Use natural language instead of exact names.

### Wrong Results

**Symptom:** Results don't match what you're looking for.

**Solutions:**

1. **Be more specific:**
   "function that validates email addresses" > "validate"

2. **Use type filters:**
   ```
   search_code("email validator")  # Code only
   search_docs("how to validate")  # Docs only
   ```

---

## Gateway Issues

### "Server not found"

**Symptom:**
```
Server not found: github
```

**Solutions:**

1. **Check config:**
   ```bash
   cat .nexus/mcp_config.json
   ```

2. **Verify server name** matches config key.

3. **Initialize config:**
   ```bash
   nexus-mcp init --from-global
   ```

### "Connection refused"

**Symptom:**
```
Error connecting to server: Connection refused
```

**Solutions:**

1. **For SSE servers:** Verify URL is accessible
   ```bash
   curl http://server-url/mcp
   ```

2. **For stdio servers:** Check command works
   ```bash
   npx -y @modelcontextprotocol/server-github
   ```

3. **Check credentials** in env configuration.

---

## Agent Issues

### Agent Not Appearing

**Symptom:** `ask_*` tool not available.

**Solutions:**

1. **Refresh agents:**
   ```
   refresh_agents()
   ```

2. **Check YAML syntax:**
   ```bash
   python -c "import yaml; yaml.safe_load(open('agents/agent.yaml'))"
   ```

3. **Verify file location:**
   Agents must be in `agents/` directory at project root.

### Agent Returns Generic Responses

**Symptom:** Agent doesn't use RAG context.

**Solutions:**

1. **Check memory config:**
   ```yaml
   memory:
     enabled: true
     rag_limit: 10
   ```

2. **Ensure content is indexed:**
   ```bash
   nexus-status
   ```

---

## Database Issues

### "Table not found"

**Symptom:**
```
LanceDB error: Table 'documents' not found
```

**Solution:**

```bash
nexus-reindex  # Rebuilds database schema
```

### Database Corruption

**Symptom:** Random errors accessing database.

**Solution:**

1. **Backup lessons:**
   ```bash
   cp -r .nexus/lessons .nexus/lessons.bak
   ```

2. **Delete database:**
   ```bash
   rm -rf ~/.local/share/nexus-dev/lancedb/*
   ```

3. **Re-index:**
   ```bash
   nexus-reindex
   ```

---

## Getting Help

### Debug Output

Run with verbose logging:
```bash
LOGLEVEL=DEBUG nexus-dev
```

### Report Issues

[GitHub Issues](https://github.com/mmornati/nexus-dev/issues)

Include:
- Nexus-Dev version (`nexus-init --version`)
- Python version (`python --version`)
- OS and version
- Full error message
- Steps to reproduce
