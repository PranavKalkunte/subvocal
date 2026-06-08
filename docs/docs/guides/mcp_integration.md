# Model Context Protocol (MCP) Integration Guide

The Subvocal SDK integrates with the Model Context Protocol (MCP), exposing intent reconstruction, gesture status buffers, and user calibration routines directly to standard AI clients like Claude Desktop.

---

## 1. MCP Server Architecture

The reference MCP server is built under `sdk/mcp/server.py`. It communicates over standard output (`stdio`) using JSON-RPC 2.0.

```
┌─────────────────┐             ┌──────────────────┐             ┌────────────────────┐
│ Claude Desktop  │ <──stdio──> │ Subvocal Server  │ <─────────> │  SubvocalPipeline  │
│  (MCP Client)   │  JSON-RPC   │   (MCP Host)     │             │    Orchestrator    │
└─────────────────┘             └──────────────────┘             └────────────────────┘
```

The server orchestrates a live `SubvocalPipeline` instance configured with a synthetic hardware stream and the pre-trained gesture models.

---

## 2. Exposed Tools and Resources

### Tools
*   **`get_pipeline_status`**: Queries active model types, hardware connection status, and policy engines.
*   **`get_buffer`**: Returns the active buffer of classified shorthand tokens.
*   **`inject_mock_token`**: Injects a custom gesture token (e.g., `clk`, `gt`, `typ`) into the pipeline buffer. Useful for testing pipelines without wearing a physical neckband.
*   **`process_phrase`**: Flushes the token buffer, runs context aggregation, queries the LLM intent reconstruction, and executes the Action.
*   **`trigger_calibration`**: Runs the online transfer learning sequence to adapt baseline classifier weights for a specific user ID.

### Resources
*   **`subvocal://intent/history`**: Exposes the trace log database, letting the AI client review recently executed actions and latency profiles.

---

## 3. Integrating with Claude Desktop

To configure Claude Desktop to load the Subvocal MCP server:

1.  Locate your Claude Desktop configuration file:
    *   **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
    *   **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
2.  Add the `subvocal` server config to the `mcpServers` node:

```json
{
  "mcpServers": {
    "subvocal": {
      "command": "python3",
      "args": [
        "-m",
        "sdk.mcp.server"
      ],
      "env": {
        "PYTHONPATH": "/absolute/path/to/subvocal/sdk",
        "OPENAI_API_KEY": "your-openai-api-key-here"
      }
    }
  }
}
```

Replace `/absolute/path/to/subvocal/sdk` with the absolute path to your local repository clone.

3.  Restart Claude Desktop. A hammer icon should appear in the text input box, indicating that the Subvocal tools are active and available for the assistant to query.
