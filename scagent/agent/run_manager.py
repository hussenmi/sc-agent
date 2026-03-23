"""
Run directory and manifest management for scagent.

Creates structured output directories with:
- Intermediate analysis files
- Figures
- Reports
- Machine-readable manifest for reproducibility
"""

import os
import json
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import getpass


@dataclass
class RunManifest:
    """Machine-readable record of an analysis run."""

    # Identity
    run_id: str = ""
    created_at: str = ""

    # Environment
    user: str = ""
    host: str = ""
    working_dir: str = ""
    scagent_version: str = ""

    # Execution
    mode: str = ""  # "library", "agent", "cli"
    model: str = ""  # Claude model if agent mode
    request: str = ""  # User's original request

    # Files
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)

    # Steps
    steps_completed: List[Dict[str, Any]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: str = "in_progress"  # "in_progress", "completed", "failed"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def save(self, path: str):
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> 'RunManifest':
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class RunManager:
    """
    Manages a structured run directory.

    Directory layout:
        run_YYYY_MM_DD_HHMMSS_<name>/
            manifest.json
            logs/
            intermediate/
                01_qc.h5ad
                02_normalized.h5ad
                ...
            figures/
                umap_clusters.png
                ...
            reports/
                summary.md
                cluster_sizes.csv
                ...
    """

    def __init__(
        self,
        base_dir: str = ".",
        run_name: Optional[str] = None,
        mode: str = "agent",
    ):
        self.base_dir = Path(base_dir)
        self.mode = mode

        # Generate run ID
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        name_part = f"_{run_name}" if run_name else ""
        self.run_id = f"run_{timestamp}{name_part}"

        # Create directory structure
        self.run_dir = self.base_dir / self.run_id
        self.dirs = {
            "root": self.run_dir,
            "logs": self.run_dir / "logs",
            "intermediate": self.run_dir / "intermediate",
            "figures": self.run_dir / "figures",
            "reports": self.run_dir / "reports",
        }

        # Initialize manifest
        self.manifest = RunManifest(
            run_id=self.run_id,
            created_at=datetime.now().isoformat(),
            user=getpass.getuser(),
            host=socket.gethostname(),
            working_dir=str(self.base_dir.absolute()),
            mode=mode,
        )

        # Step counter for ordering
        self._step_counter = 0

    def create(self):
        """Create the run directory structure."""
        for name, path in self.dirs.items():
            path.mkdir(parents=True, exist_ok=True)

        # Save initial manifest
        self._save_manifest()

        return self

    def _save_manifest(self):
        """Save manifest to disk."""
        manifest_path = self.run_dir / "manifest.json"
        self.manifest.save(str(manifest_path))

    def append_log(self, message: str, filename: str = "agent.log"):
        """Append a line to a run log file."""
        log_path = self.dirs["logs"] / filename
        timestamp = datetime.now().isoformat()
        with open(log_path, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        return str(log_path)

    def set_request(self, request: str):
        """Set the user's original request."""
        self.manifest.request = request
        self._save_manifest()

    def set_model(self, model: str):
        """Set the Claude model being used."""
        self.manifest.model = model
        self._save_manifest()

    def set_version(self, version: str):
        """Set scagent version."""
        self.manifest.scagent_version = version
        self._save_manifest()

    def add_input(self, path: str):
        """Register an input file."""
        self.manifest.input_files.append(str(path))
        self._save_manifest()

    def add_output(self, path: str):
        """Register an output file."""
        self.manifest.output_files.append(str(path))
        self._save_manifest()

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.manifest.warnings.append(warning)
        self._save_manifest()

    def get_intermediate_path(self, name: str, ext: str = "h5ad") -> str:
        """
        Get path for an intermediate file with ordering prefix.

        Example: get_intermediate_path("qc") -> "intermediate/01_qc.h5ad"
        """
        self._step_counter += 1
        filename = f"{self._step_counter:02d}_{name}.{ext}"
        return str(self.dirs["intermediate"] / filename)

    def get_figure_path(self, name: str, ext: str = "png") -> str:
        """Get path for a figure."""
        return str(self.dirs["figures"] / f"{name}.{ext}")

    def get_report_path(self, name: str, ext: str = "csv") -> str:
        """Get path for a report file."""
        return str(self.dirs["reports"] / f"{name}.{ext}")

    def write_text_report(self, name: str, content: str, ext: str = "md") -> str:
        """Write a text report into the reports directory and register it."""
        path = self.dirs["reports"] / f"{name}.{ext}"
        with open(path, "w") as f:
            f.write(content)
        self.add_output(str(path))
        return str(path)

    def write_json_report(self, name: str, payload: Dict[str, Any]) -> str:
        """Write a JSON report into the reports directory and register it."""
        path = self.dirs["reports"] / f"{name}.json"
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        self.add_output(str(path))
        return str(path)

    def log_step(
        self,
        tool: str,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Log a completed analysis step."""
        step = {
            "step": len(self.manifest.steps_completed) + 1,
            "tool": tool,
            "timestamp": datetime.now().isoformat(),
            "input_path": input_path,
            "output_path": output_path,
            "parameters": parameters or {},
        }

        if result:
            # Include key metrics, not full result
            step["metrics"] = {
                k: v for k, v in result.items()
                if k in ["before", "after", "n_clusters", "n_hvg", "doublet_rate", "status"]
            }

        self.manifest.steps_completed.append(step)

        if output_path:
            self.add_output(output_path)

        self._save_manifest()

    def complete(self, summary: str = ""):
        """Mark run as completed and save final manifest."""
        self.manifest.status = "completed"
        self._save_manifest()

        # Write summary report
        summary_path = self.dirs["reports"] / "summary.md"
        with open(summary_path, 'w') as f:
            f.write(f"# Run Summary: {self.run_id}\n\n")
            f.write(f"**Status:** {self.manifest.status}\n")
            f.write(f"**Created:** {self.manifest.created_at}\n")
            f.write(f"**User:** {self.manifest.user}@{self.manifest.host}\n\n")

            if self.manifest.request:
                f.write(f"## Request\n\n{self.manifest.request}\n\n")

            f.write("## Steps Completed\n\n")
            for step in self.manifest.steps_completed:
                f.write(f"- **{step['tool']}**")
                if step.get('output_path'):
                    f.write(f" → `{Path(step['output_path']).name}`")
                f.write("\n")

            if summary:
                f.write(f"\n## Summary\n\n{summary}\n")

            if self.manifest.warnings:
                f.write("\n## Warnings\n\n")
                for w in self.manifest.warnings:
                    f.write(f"- {w}\n")

        return str(summary_path)

    def fail(self, error: str):
        """Mark run as failed."""
        self.manifest.status = "failed"
        self.manifest.errors.append(error)
        self._save_manifest()


def create_run(
    base_dir: str = ".",
    run_name: Optional[str] = None,
    mode: str = "agent",
) -> RunManager:
    """
    Create and initialize a new run directory.

    Parameters
    ----------
    base_dir : str
        Directory to create run folder in.
    run_name : str, optional
        Name suffix for the run directory.
    mode : str
        Execution mode: "agent", "library", "cli".

    Returns
    -------
    RunManager
        Initialized run manager.
    """
    manager = RunManager(base_dir=base_dir, run_name=run_name, mode=mode)
    manager.create()
    return manager
