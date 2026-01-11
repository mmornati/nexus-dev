# Nexus-Dev Demo: Counting Closed GitHub Issues

This document details the steps taken by the AI agent to count closed issues in the `mmornati/nexus-dev` repository using the Nexus-Dev MCP gateway.

## Step 1: Verification of Available Tools
First, the agent verified that the GitHub MCP server was active and available through the Nexus-Dev gateway.

**Tool:** `list_servers`
**Reasoning:** Before attempting to use any GitHub tools, it is necessary to confirm that the GitHub MCP server is correctly connected and recognized by the Nexus-Dev system.
**Result:** Verified `github` server is active.

## Step 2: Tool Discovery
The agent searched for the appropriate tool to list GitHub issues.

**Tool:** `search_tools(query="list issues", server="github")`
**Reasoning:** The agent needed to find the specific function signature for listing issues to ensure the correct arguments (like `state="CLOSED"`) were provided.
**Result:** Found `github.list_issues` with the following relevant parameters:
- `owner` (string): Repository owner
- `repo` (string): Repository name
- `state` (string): Filter by state (`OPEN`, `CLOSED`)

## Step 3: Execution
The agent invoked the discovered tool to retrieve the data.

**Tool:** `invoke_tool(server="github", tool="list_issues", arguments={"owner": "mmornati", "repo": "nexus-dev", "state": "CLOSED"})`
**Reasoning:** With the tool and parameters identified, the agent executed the command to fetch the actual data.
**Result:** The tool returned a JSON response containing a list of issues.

## Step 4: Analysis
The agent processed the JSON output to count the number of items in the returned list.

**Reasoning:** The tool returns the raw data (lists of issues). The agent's role is to interpret this data to answer the user's specific question ("Count the closed issues").
**Final Answer:** There are **23** closed issues in the `mmornati/nexus-dev` repository.
