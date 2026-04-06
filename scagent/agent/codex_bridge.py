"""Codex CLI bridge for ChatGPT-login-backed SCAgent runs.

This backend intentionally goes through the official ``codex`` command instead
of reading Codex auth tokens directly. Codex handles ChatGPT/API-key auth and
SCAgent only asks it for the next high-level tool decision.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


CODEX_DECISION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": ["tool_call", "final"]},
        "thought": {"type": "string"},
        "tool_name": {"type": ["string", "null"]},
        "tool_input_json": {"type": ["string", "null"]},
        "content": {"type": ["string", "null"]},
    },
    "required": ["kind", "thought", "tool_name", "tool_input_json", "content"],
}


class CodexCLIError(RuntimeError):
    """Raised when the Codex CLI bridge cannot complete a request."""


class CodexCLIClient:
    """Thin wrapper around ``codex exec`` and ``codex login``."""

    def __init__(
        self,
        command: Optional[str] = None,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.command = command or os.environ.get("SCAGENT_CODEX_COMMAND", "codex")
        self.cwd = cwd or os.getcwd()
        self.model = model
        self.timeout = timeout or int(os.environ.get("SCAGENT_CODEX_TIMEOUT", "600"))

        if shutil.which(self.command) is None and not Path(self.command).exists():
            raise CodexCLIError(
                "Codex CLI was not found. Install Codex or set SCAGENT_CODEX_COMMAND."
            )

    def login_status(self) -> str:
        """Return the Codex login status text."""
        result = subprocess.run(
            [self.command, "login", "status"],
            cwd=self.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
        )
        output = (result.stdout or result.stderr).strip()
        if result.returncode != 0:
            raise CodexCLIError(output or "Codex login status failed.")
        return output

    def complete_json(self, prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Run ``codex exec`` and parse the final JSON response."""
        with tempfile.TemporaryDirectory(prefix="scagent_codex_") as tmpdir:
            tmp = Path(tmpdir)
            schema_path = tmp / "schema.json"
            output_path = tmp / "last_message.json"
            schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")

            cmd = [
                self.command,
                "exec",
                "--ephemeral",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--output-schema",
                str(schema_path),
                "--output-last-message",
                str(output_path),
            ]
            if self.model:
                cmd.extend(["--model", self.model])
            cmd.append("-")

            result = subprocess.run(
                cmd,
                input=prompt,
                cwd=self.cwd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                details = self._summarize_failure(result.stdout, result.stderr)
                raise CodexCLIError(details or f"Codex exited with status {result.returncode}.")

            text = ""
            if output_path.exists():
                text = output_path.read_text(encoding="utf-8").strip()
            if not text:
                text = (result.stdout or "").strip()
            return self._parse_json(text)

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        if not text:
            raise CodexCLIError("Codex returned an empty response.")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CodexCLIError(f"Codex returned non-JSON output: {text[:1000]}") from exc
        if not isinstance(data, dict):
            raise CodexCLIError("Codex returned JSON, but not a JSON object.")
        return data

    @staticmethod
    def _summarize_failure(stdout: str, stderr: str) -> str:
        """Keep Codex failures useful without echoing the entire prompt."""
        combined = "\n".join(part for part in [stderr, stdout] if part).strip()
        if not combined:
            return ""
        lines = combined.splitlines()
        error_lines = [
            line for line in lines
            if "ERROR:" in line or "invalid_request_error" in line or "error" in line.lower()
        ]
        if error_lines:
            return "\n".join(error_lines[-8:])
        return "\n".join(lines[-20:])
