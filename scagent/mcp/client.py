"""
Persistent MCP client for scagent.

Reads .mcp.json (same format as Claude Code), connects to each configured
MCP server via stdio, fetches tool schemas, and exposes them for the agent.
All server connections are kept alive in a background asyncio event loop so
tool calls don't pay process-spawn overhead on every call.
"""

import asyncio
import json
import logging
import os
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _find_mcp_config(start_dir: Optional[str] = None) -> Optional[Path]:
    """Walk up from start_dir looking for .mcp.json, same as Claude Code does."""
    search = Path(start_dir or os.getcwd()).resolve()
    for directory in [search, *search.parents]:
        candidate = directory / ".mcp.json"
        if candidate.exists():
            return candidate
    # Also check SCAGENT_MCP_CONFIG env var as explicit override
    env_path = os.environ.get("SCAGENT_MCP_CONFIG")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    return None


class MCPClientManager:
    """
    Manages persistent connections to MCP servers and exposes their tools.

    Usage:
        manager = MCPClientManager.from_config()   # auto-finds .mcp.json
        manager.start()                            # connect to all servers
        schemas = manager.tool_schemas             # Anthropic-format tool dicts
        result = manager.call_tool("bc_get_panglaodb_marker_genes", {"species": "Hs", "cell_type": "T cells"})
        manager.stop()
    """

    def __init__(self, server_configs: Dict[str, Dict[str, Any]]):
        """
        Parameters
        ----------
        server_configs : dict
            Contents of the mcpServers block from .mcp.json.
            Each key is a server name; value has 'command', optional 'args', optional 'env'.
        """
        self._server_configs = server_configs
        self._tool_schemas: List[Dict[str, Any]] = []
        self._tool_to_server: Dict[str, str] = {}

        # Async state — managed by the background thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._sessions: Dict[str, Any] = {}   # server_name -> ClientSession
        self._exit_stacks: Dict[str, AsyncExitStack] = {}
        self._started = False
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, config_path: Optional[str] = None, start_dir: Optional[str] = None) -> "MCPClientManager":
        """
        Create a manager by reading .mcp.json.

        Parameters
        ----------
        config_path : str, optional
            Explicit path to .mcp.json. If None, searches up from start_dir.
        start_dir : str, optional
            Directory to start searching from (default: cwd).
        """
        if config_path:
            path = Path(config_path)
        else:
            path = _find_mcp_config(start_dir)

        if path is None or not path.exists():
            logger.info("No .mcp.json found — MCP tools will not be available.")
            return cls({})

        with open(path) as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})
        logger.info("MCP config loaded from %s (%d servers)", path, len(servers))
        return cls(servers)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background event loop and connect to all configured servers."""
        if self._started:
            return
        if not self._server_configs:
            self._started = True
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever,
            name="scagent-mcp-loop",
            daemon=True,
        )
        self._thread.start()

        future = asyncio.run_coroutine_threadsafe(self._connect_all(), self._loop)
        try:
            future.result(timeout=60)
        except Exception as e:
            logger.warning("MCP startup error: %s", e)
        self._started = True

    def stop(self) -> None:
        """Gracefully shut down all MCP server connections."""
        if not self._started or self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._cleanup_all(), self._loop)
        try:
            future.result(timeout=15)
        except Exception as e:
            logger.debug("MCP cleanup error: %s", e)
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        self._started = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Return Anthropic-format tool schemas for all tools from all connected servers."""
        return list(self._tool_schemas)

    @property
    def connected_servers(self) -> List[str]:
        return list(self._sessions.keys())

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tool_to_server

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
        """
        Synchronously call an MCP tool and return the parsed result.

        Returns a dict that can be JSON-serialised and returned to the LLM.
        Raises RuntimeError on failure.
        """
        if not self._started:
            raise RuntimeError("MCPClientManager has not been started. Call start() first.")

        server_name = self._tool_to_server.get(tool_name)
        if not server_name:
            raise ValueError(f"Unknown MCP tool: '{tool_name}'. Available: {list(self._tool_to_server)}")

        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(server_name, tool_name, arguments),
            self._loop,
        )
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal async helpers (all run on the background loop)
    # ------------------------------------------------------------------

    async def _connect_all(self) -> None:
        for name, cfg in self._server_configs.items():
            try:
                await self._connect_server(name, cfg)
                logger.info("MCP server '%s': connected", name)
            except Exception as e:
                logger.warning("MCP server '%s': failed to connect (%s)", name, e)

    async def _connect_server(self, name: str, cfg: Dict[str, Any]) -> None:
        import io
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client, StdioServerParameters

        command = cfg["command"]
        args = cfg.get("args", [])
        env_overrides = cfg.get("env", {})

        # Merge server env overrides into current environment
        merged_env = {**os.environ, **env_overrides} if env_overrides else None

        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=merged_env,
        )

        exit_stack = AsyncExitStack()
        # Open /dev/null as errlog to suppress server startup banners from leaking to terminal
        devnull = exit_stack.enter_context(open(os.devnull, "w"))
        read, write = await exit_stack.enter_async_context(stdio_client(server_params, errlog=devnull))
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        self._sessions[name] = session
        self._exit_stacks[name] = exit_stack

        # Fetch tool schemas and register them
        tools_result = await session.list_tools()
        for tool in tools_result.tools:
            self._tool_to_server[tool.name] = name
            # Convert MCP tool schema to Anthropic format
            input_schema = tool.inputSchema or {"type": "object", "properties": {}}
            # Remove $schema key that MCP sometimes adds — Anthropic rejects it
            input_schema = {k: v for k, v in input_schema.items() if k != "$schema"}
            self._tool_schemas.append({
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": input_schema,
            })

        logger.info(
            "MCP server '%s': %d tools registered",
            name,
            len(tools_result.tools),
        )

    async def _call_tool_async(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments)

        # Parse the MCP result content into a plain dict for the agent
        if result.isError:
            error_text = " ".join(
                item.text for item in result.content if hasattr(item, "text")
            )
            return {"status": "error", "tool": tool_name, "message": error_text}

        # Collect text content items; try to parse as JSON, fall back to plain text
        parts = []
        for item in result.content:
            if hasattr(item, "text"):
                try:
                    parts.append(json.loads(item.text))
                except (json.JSONDecodeError, ValueError):
                    parts.append({"text": item.text})

        if len(parts) == 1:
            payload = parts[0]
        elif len(parts) > 1:
            payload = {"results": parts}
        else:
            payload = {}

        # Wrap in a standard envelope if it's not already one
        if not isinstance(payload, dict):
            payload = {"value": payload}

        payload.setdefault("status", "ok")
        payload.setdefault("tool", tool_name)
        return payload

    async def _cleanup_all(self) -> None:
        for name, stack in list(self._exit_stacks.items()):
            try:
                await stack.aclose()
            except Exception as e:
                logger.debug("Cleanup error for server '%s': %s", name, e)
        self._sessions.clear()
        self._exit_stacks.clear()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "started" if self._started else "stopped"
        return (
            f"MCPClientManager({status}, servers={list(self._server_configs)}, "
            f"tools={len(self._tool_schemas)})"
        )
