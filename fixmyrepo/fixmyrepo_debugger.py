"""
Autonomous Debug Loop for openfold - Modal Sandbox Edition.

Uses Claude Agent SDK to automatically fix errors until prediction succeeds.

┌─────────────────────────────────────────────────────────────────────────────┐
│ TWO-LEVEL ERROR HANDLING:                                                    │
│ 1. Runtime errors (code bugs) → Claude fixes code → re-run in same sandbox  │
│ 2. Environment errors (CANNOT_FIX) → Claude fixes THIS file → full rebuild  │
└─────────────────────────────────────────────────────────────────────────────┘

Usage:
    uv run python fixmyrepo_debugger.py \
        /path/to/input.fasta \
        /path/to/templates/ \
        --output_dir /path/to/output/ \
        --model_device cuda:0 \
        --config_preset model_1

Requirements:
    uv add modal claude-agent-sdk
"""

import modal
import asyncio
import argparse
import sys
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage, AssistantMessage, TextBlock


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - Auto-generated from repository analysis
# ════════════════════════════════════════════════════════════════════════════

APP_NAME = "openfold-autonomous"
REPO_NAME = "openfold"
WORKING_DIR = "/root/openfold"
MAX_RETRIES = 10

# GPU configuration
GPU_REQUIRED = True
GPU_TYPE = "A10G"

# Timeout configuration (in seconds)
SANDBOX_TIMEOUT_MINUTES = 60  # Default: 60
PREDICTION_TIMEOUT_SECONDS = 1800  # Default: 1800

# Checkpoint configuration
CHECKPOINT_URL = None
CHECKPOINT_NAMES = []  # Valid checkpoint options
DEFAULT_CHECKPOINT = None
CHECKPOINT_EXTENSION = ""  # e.g., ".ckpt", ".pt", ".pth", or ""
CHECKPOINT_AUTO_DOWNLOAD = False  # True if framework downloads weights automatically (ColabFold, HuggingFace, etc.)

# File sync configuration
EXCLUDE_PATTERNS = [".venv/", "venv/", "__pycache__/", ".git/", "*.pyc", ".pytest_cache/", "node_modules/", ".mypy_cache/", ".tox/", "dist/", "build/", ".eggs/", "*.egg-info/"]
SYNC_PATTERNS = ["**/*.py", "*.yaml", "*.toml"]

# Optional parameters (set to None if not used by this repo)
HAS_N_SAMPLES = False

# ════════════════════════════════════════════════════════════════════════════
# SETUP METHOD CONFIGURATION (NEW - handles diverse repo patterns)
# ════════════════════════════════════════════════════════════════════════════
# Setup method determines how the environment is built:
#   - docker: Use pre-built Docker image (FASTEST & PREFERRED - use when docker_image is set)
#   - deprecated: Repo is deprecated, error with redirect
#   - external_docs: Setup info at external URL
#   - setup_script: Run setup.sh or install.sh
#   - conda_env_yaml: Use environment.yaml file
#   - conda_requirements: conda create + pip install -r requirements.txt
#   - pip_install: Simple pip install

SETUP_METHOD = "conda_env_yaml"

ENVIRONMENT_FILE = "environment.yml"

# Optional: Command to install repo itself (e.g., "pip install -e .[cuda]")
REPO_INSTALL_CMD = None  # None or "pip install -e ."

# Optional: Command to download checkpoints
CHECKPOINT_DOWNLOAD_CMD = "bash scripts/download_openfold_params.sh openfold/resources"  # None or "bash get_model_params.sh ./model_params"

# Optional: Special pip commands with flags (e.g., PyTorch Geometric)
SPECIAL_PIP_COMMANDS = ["pip install deepspeed==0.14.5 --no-build-isolation", "pip install flash-attn --no-build-isolation"]  # [] or ["pip install pyg_lib -f https://..."]

# Conda packages that uv/pip MUST NOT upgrade (prevents ABI conflicts)
# These will be pinned via --constraint when running uv pip install
CONDA_PROTECTED = ["torch", "numpy", "scipy", "pandas", "biopython"]  # e.g., ["torch", "numpy", "scipy"]

# ─── Legacy/Fallback Configuration ───
# These are used when SETUP_METHOD is conda_requirements or pip_install

USE_PREBUILT_IMAGE = False
DOCKER_IMAGE = None

# Conda packages (scientific/compiled - need conda for prebuilt binaries)
CONDA_PACKAGES = []

# Pip packages (installed via uv for speed)
PIP_PACKAGES = []

# Conda channels
CONDA_CHANNELS = ""

# ────────────────────────────────────────────────────────────────────────────
# PREDICTION COMMAND TEMPLATE
# ────────────────────────────────────────────────────────────────────────────
# NOTE ON TEMPLATE SYNTAX:
#   - {{PREDICTION_CMD}} is a Jinja2 variable (replaced during template rendering)
#   - The PREDICTION_CMD VALUE contains Python format strings like {protein}, {checkpoint}
#   - These Python format strings are filled at RUNTIME by run_prediction()
#
# Example: If PREDICTION_CMD = "python run.py --input {protein} --ckpt {checkpoint}"
#   - Jinja2 replaces {{PREDICTION_CMD}} with the literal string
#   - At runtime, .format(protein="/path/to/file", checkpoint="/path/to/ckpt") fills values
#
# OUTPUT ENABLE FLAGS:
#   - Some outputs require CLI flags to be enabled (e.g., --save_score 1)
#   - These are collected from OUTPUTS[].enable_flag and appended to the command
#   - Empty enable_flags are skipped (output is always produced)
# ────────────────────────────────────────────────────────────────────────────
PREDICTION_CMD = """python run_pretrained_openfold.py {input_fasta} {template_mmcif_dir} --output_dir {output_dir} --model_device {model_device} --config_preset {config_preset} --jax_param_path {jax_param_path} --openfold_checkpoint_path {openfold_checkpoint_path}"""

# Repository-specific notes for Claude (helps Claude understand repo quirks)
REPO_NOTES = """OpenFold requires conda environment setup from environment.yml, followed by running scripts/install_third_party_dependencies.sh which downloads stereo_chemical_props.txt, decompresses test data, runs python setup.py install to compile CUDA kernels, clones CUTLASS v3.6.0, and sets environment variables. Model weights must be downloaded separately using provided scripts (AlphaFold params, OpenFold params, or SoloSeq params). The setup.py compiles custom CUDA attention kernels that require CUDA 12.1+. Environment uses PyTorch 2.5 with CUDA 12.4, includes bioinformatics tools (hmmer, hhsuite, kalign2), and pip packages (deepspeed, flash-attn, dllogger from git). Supports FlashAttention and DeepSpeed DS4Sci_EvoformerAttention kernel for memory-efficient training/inference. Can handle sequences with 4000+ residues on A100. While a Dockerfile exists for local building, no pre-built Docker image is published to any registry."""


# ════════════════════════════════════════════════════════════════════════════
# CLI MANIFEST (backup for automated arg parsing)
# ════════════════════════════════════════════════════════════════════════════
# This JSON defines the CLI structure. Used as fallback if --help parsing fails.
# Format: Single-line JSON comment for safe parsing with json.loads()
#
# FIXMYREPO_CLI: {"positional_args": ["input_fasta", "template_mmcif_dir"], "flag_args": {"output_dir": "--output_dir", "model_device": "--model_device", "config_preset": "--config_preset", "jax_param_path": "--jax_param_path", "openfold_checkpoint_path": "--openfold_checkpoint_path", "uniref90_database_path": "--uniref90_database_path", "mgnify_database_path": "--mgnify_database_path", "pdb70_database_path": "--pdb70_database_path", "uniclust30_database_path": "--uniclust30_database_path", "bfd_database_path": "--bfd_database_path", "jackhmmer_binary_path": "--jackhmmer_binary_path", "hhblits_binary_path": "--hhblits_binary_path", "hhsearch_binary_path": "--hhsearch_binary_path", "kalign_binary_path": "--kalign_binary_path"}, "defaults": {"checkpoint": null, "output": "/root/output/"}}
#
# ════════════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PredictionResult:
    """Result of a prediction run."""
    success: bool
    stdout: str
    stderr: str
    returncode: int


@dataclass
class InputFile:
    """Definition of an input file."""
    name: str
    local_path: Path
    remote_path: str


# ════════════════════════════════════════════════════════════════════════════
# MODAL SANDBOX MANAGER
# ════════════════════════════════════════════════════════════════════════════

class ModalSandboxManager:
    """
    Manages a long-lived Modal sandbox for iterative debugging.

    Key features:
    - Keeps sandbox alive between prediction attempts (no rebuild per retry)
    - Syncs local code changes into running sandbox
    - Handles checkpoint downloading and caching
    - Supports both pre-built Docker images and from-scratch builds
    """

    def __init__(self):
        self.app = modal.App.lookup(APP_NAME, create_if_missing=True)
        self.local_dir = Path(__file__).parent.parent.resolve()  # Go up from fixmyrepo/ to repo root
        self.sandbox = None
        self.python_cmd = None  # Detected after sandbox creation by _detect_python()
        self._build_image()
        self.volume = modal.Volume.from_name(f"{APP_NAME}-data", create_if_missing=True)
        # Shared cache volume for model weights, pip cache, etc. (persists across ALL repos)
        self.cache_volume = modal.Volume.from_name("fixmyrepo-cache", create_if_missing=True)

    def _build_image(self):
        """Build the Modal image based on SETUP_METHOD configuration."""

        # ─── CONDA ENVIRONMENT FILE MODE ───
        # Use the repo's environment.yaml directly
        print(f"Using conda environment file: {ENVIRONMENT_FILE}")

        # Build special pip commands string
        special_pip_cmds = SPECIAL_PIP_COMMANDS if SPECIAL_PIP_COMMANDS else []

        # ═══════════════════════════════════════════════════════════════════════
        # BUILD ISOLATION FIX
        # ═══════════════════════════════════════════════════════════════════════
        # Some packages (flash-attn, deepspeed, xformers) fail when pip builds them
        # in isolation because they import torch in setup.py. When these packages
        # are in special_pip_commands with --no-build-isolation, we need to:
        # 1. Remove them from environment.yml's pip section BEFORE micromamba runs
        # 2. Install them AFTER micromamba using special_pip_commands
        # ═══════════════════════════════════════════════════════════════════════

        # Extract package names from special_pip_commands that use --no-build-isolation
        packages_to_skip = []
        for cmd in special_pip_cmds:
            if "--no-build-isolation" in cmd:
                # Extract package name from "pip install flash-attn --no-build-isolation"
                parts = cmd.replace("pip install", "").replace("--no-build-isolation", "").strip().split()
                if parts:
                    # Get the package name (first part, without version specifiers)
                    pkg = parts[0].split("==")[0].split(">=")[0].split("<=")[0].split("[")[0]
                    packages_to_skip.append(pkg)

        self.image = (
            modal.Image.debian_slim()
            .apt_install("curl", "ca-certificates", "git", "wget", "bzip2", "python3", "python3-pip")
            .run_commands(
                "pip3 install pyyaml",
            )
            .run_commands(
                # Install micromamba
                "curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr bin/micromamba",
                "micromamba shell init -s bash --root-prefix /opt/conda",
                # Install uv
                "curl -LsSf https://astral.sh/uv/install.sh | sh",
            )
            .env({
                "MAMBA_ROOT_PREFIX": "/opt/conda",
                "PATH": "/opt/conda/envs/openfold-autonomous/bin:/opt/conda/bin:/root/.local/bin:$PATH",
                "LD_LIBRARY_PATH": "/opt/conda/envs/openfold-autonomous/lib:$LD_LIBRARY_PATH",
            })
            # copy=True required because run_commands follows and needs access to environment.yaml
            .add_local_dir(
                str(self.local_dir),
                WORKING_DIR,
                ignore=EXCLUDE_PATTERNS,
                copy=True,
            )
        )

        # If we have packages to skip, modify environment.yml before running micromamba
        if packages_to_skip:
            skip_list = ",".join(f'"{p}"' for p in packages_to_skip)
            print(f"Removing packages from environment.yml pip section (will install separately with --no-build-isolation): {packages_to_skip}")

            # Python script to remove packages from environment.yml pip section
            # Use base64 encoding to avoid shell quote escaping issues
            modify_env_script = f'''
import yaml
import re

env_file = "{WORKING_DIR}/{ENVIRONMENT_FILE}"
with open(env_file, "r") as f:
    env = yaml.safe_load(f)

packages_to_skip = [{skip_list}]
packages_to_skip_normalized = [p.lower().replace("-", "_") for p in packages_to_skip]

# Find and modify the pip section
new_deps = []
for dep in env.get("dependencies", []):
    if isinstance(dep, dict) and "pip" in dep:
        # Filter out packages we want to skip
        filtered_pip = []
        for pkg in dep["pip"]:
            pkg_name = pkg.split("==")[0].split(">=")[0].split("<=")[0].split("[")[0].strip()
            pkg_normalized = pkg_name.lower().replace("-", "_")
            if pkg_normalized not in packages_to_skip_normalized:
                filtered_pip.append(pkg)
            else:
                print("  Removed from pip section: " + pkg)
        if filtered_pip:
            new_deps.append({{"pip": filtered_pip}})
    else:
        new_deps.append(dep)

env["dependencies"] = new_deps

with open(env_file, "w") as f:
    yaml.dump(env, f, default_flow_style=False)
print("Modified " + env_file + " - removed " + str(len(packages_to_skip)) + " packages from pip section")
'''
            # Write script to temp file and execute it to avoid shell escaping issues
            import base64
            script_b64 = base64.b64encode(modify_env_script.encode()).decode()
            self.image = self.image.run_commands(
                f'echo "{script_b64}" | base64 -d > /tmp/modify_env.py && python3 /tmp/modify_env.py'
            )

        # Now run micromamba with (potentially modified) environment file
        self.image = self.image.run_commands(
            f"micromamba create -n openfold-autonomous -f {WORKING_DIR}/{ENVIRONMENT_FILE} -y",
        )

        # Add special pip commands (e.g., --no-build-isolation packages, PyTorch Geometric with -f URL)
        for cmd in special_pip_cmds:
            self.image = self.image.run_commands(
                f"cd {WORKING_DIR} && {cmd}"
            )

        # Add repo install command if needed
        if REPO_INSTALL_CMD:
            self.image = self.image.run_commands(
                f"cd {WORKING_DIR} && {REPO_INSTALL_CMD}"
            )

    # ════════════════════════════════════════════════════════════════════════════
    # PRE-FLIGHT VERIFICATION (Dec 2025)
    # ════════════════════════════════════════════════════════════════════════════
    # These methods detect the working Python command and verify the environment
    # after sandbox creation. This prevents "executable 'python' not found" errors
    # that fell through the cracks between meta-loop and internal debug loop.
    # ════════════════════════════════════════════════════════════════════════════

    def _detect_python(self) -> str:
        """
        Detect the working Python executable in the sandbox.

        Tries multiple Python paths common in Docker/conda environments:
        1. micromamba run -n <env> python (micromamba envs) - TRIED FIRST
        2. Direct paths in conda environment bin directories
        3. python3 / python in PATH
        4. /usr/bin/python3 (system Python)
        5. Dynamically discovered conda/micromamba environments

        For pre-built Docker images (e.g., rbgcsail/diffdock), the environment
        name often matches the repo name in lowercase (e.g., "diffdock").

        Stores result in self.python_cmd for use by _exec_python().
        Raises RuntimeError with INFRASTRUCTURE_ERROR prefix if no Python found.
        """
        # Derive likely environment name from repo name (for pre-built images)
        repo_env_name = REPO_NAME.lower().replace("-", "_").replace(" ", "_")

        # Dynamically discover conda/micromamba environments
        discovered_envs = []
        try:
            # Try micromamba env list first (parses output for env names)
            p = self.sandbox.exec("bash", "-c", "micromamba env list 2>/dev/null | grep -E '^[a-zA-Z]' | awk '{print $1}'")
            stdout = p.stdout.read().strip()
            p.wait()
            if p.returncode == 0 and stdout:
                discovered_envs = [e.strip() for e in stdout.split('\n') if e.strip()]
        except Exception:
            pass

        if not discovered_envs:
            try:
                # Fallback: list /opt/conda/envs/ directory
                p = self.sandbox.exec("bash", "-c", "ls /opt/conda/envs/ 2>/dev/null")
                stdout = p.stdout.read().strip()
                p.wait()
                if p.returncode == 0 and stdout:
                    discovered_envs = [e.strip() for e in stdout.split('\n') if e.strip()]
            except Exception:
                pass

        # Try direct paths FIRST (most reliable for micromamba environments)
        # The PATH environment variable is set to include the conda env bin directory
        python_candidates = [
            f"/opt/conda/envs/{APP_NAME}/bin/python",
            f"/opt/conda/envs/{repo_env_name}/bin/python",
            "python3",
            "python",
            "/usr/bin/python3",
            "/usr/bin/python",
            "/opt/conda/bin/python",
            "/opt/conda/bin/python3",
            "/root/.local/bin/python3",
            "/root/.local/bin/python",
        ]

        # Add discovered environment paths to candidates
        for env_name in discovered_envs:
            path = f"/opt/conda/envs/{env_name}/bin/python"
            if path not in python_candidates:
                python_candidates.insert(2, path)  # Try discovered envs early

        print(f"Trying {len(python_candidates)} direct path candidates...")

        for cmd in python_candidates:
            try:
                p = self.sandbox.exec("bash", "-c", f"{cmd} --version")
                stdout = p.stdout.read().strip()
                p.wait()
                if p.returncode == 0:
                    print(f"Detected Python: {cmd} ({stdout})")
                    self.python_cmd = cmd
                    return cmd
            except Exception:
                continue

        # Fallback: Try micromamba run for each discovered environment
        # This may fail if micromamba shell isn't initialized properly
        envs_to_try = [APP_NAME, repo_env_name] + discovered_envs
        print(f"Python detection: trying {len(envs_to_try)} micromamba environments...")

        for env_name in envs_to_try:
            try:
                p = self.sandbox.exec(
                    "bash", "-c",
                    f"micromamba run -n {env_name} python --version"
                )
                stdout = p.stdout.read().strip()
                p.wait()
                if p.returncode == 0:
                    cmd = f"micromamba run -n {env_name} python"
                    print(f"Detected Python: {cmd} ({stdout})")
                    self.python_cmd = cmd
                    return cmd
            except Exception:
                continue

        raise RuntimeError(
            f"INFRASTRUCTURE_ERROR: No Python executable found in sandbox. "
            f"Tried micromamba environments {envs_to_try} and direct paths {python_candidates}. "
            f"Check that the Docker image or conda environment has Python installed."
        )

    def _verify_environment(self) -> None:
        """
        Verify the sandbox environment is properly configured.

        Checks:
        1. Python can import basic modules (sys, os)
        2. Working directory exists and is accessible

        Raises RuntimeError with INFRASTRUCTURE_ERROR prefix on failure.
        """
        # Test basic Python functionality
        test_script = "import sys, os; print(f'Python {sys.version_info.major}.{sys.version_info.minor} OK')"
        try:
            p = self._exec_python("-c", test_script)
            p.wait()
            if p.returncode != 0:
                stderr = p.stderr.read()
                raise RuntimeError(
                    f"INFRASTRUCTURE_ERROR: Python environment check failed. "
                    f"stderr: {stderr}"
                )
            result = p.stdout.read().strip()
            print(f"Environment verified: {result}")
        except Exception as e:
            if "INFRASTRUCTURE_ERROR" in str(e):
                raise
            raise RuntimeError(
                f"INFRASTRUCTURE_ERROR: Failed to verify Python environment: {e}"
            )

        # Verify working directory exists
        p = self.sandbox.exec("bash", "-c", f"test -d {WORKING_DIR} && echo OK || echo MISSING")
        p.wait()
        result = p.stdout.read().strip()
        if result != "OK":
            raise RuntimeError(
                f"INFRASTRUCTURE_ERROR: Working directory {WORKING_DIR} does not exist in sandbox."
            )

    def _exec_python(self, *args) -> "modal.sandbox.ContainerProcess":
        """
        Execute Python with detected command.

        Handles both simple commands (python3) and compound commands
        (micromamba run -n env python).

        Args:
            *args: Arguments to pass to Python (e.g., "-c", "print('hello')")

        Returns:
            ContainerProcess from sandbox.exec()
        """
        if not self.python_cmd:
            raise RuntimeError(
                "INFRASTRUCTURE_ERROR: python_cmd not set. Call _detect_python() first."
            )

        # Handle compound commands like "micromamba run -n env python"
        if " " in self.python_cmd:
            # Compound command - use bash -c
            full_cmd = f"{self.python_cmd} {' '.join(args)}"
            return self.sandbox.exec("bash", "-c", full_cmd)
        else:
            # Simple command - direct exec
            return self.sandbox.exec(self.python_cmd, *args)

    def create(self, timeout_minutes: int = None):
        """
        Create sandbox (keeps alive for entire debugging loop).

        The sandbox persists between prediction attempts, allowing:
        - Code changes to be synced without rebuilding
        - Checkpoints to be cached
        - Faster iteration cycles
        """
        timeout_minutes = timeout_minutes or SANDBOX_TIMEOUT_MINUTES

        print(f"Creating Modal Sandbox with {GPU_TYPE} GPU...")
        print("Using from-scratch build with micromamba + uv")

        with modal.enable_output():
            self.sandbox = modal.Sandbox.create(
                image=self.image,
                gpu=GPU_TYPE,
                # NOTE: Volume mounts disabled due to Modal sandbox crash bug (Dec 2025)
                # The sandbox crashes immediately when volumes are mounted to any path
                # Caching disabled until this is resolved - model weights downloaded each run
                # volumes={
                #     "/mnt/data": self.volume,
                #     "/mnt/cache": self.cache_volume,
                # },
                workdir=WORKING_DIR,
                timeout=timeout_minutes * 60,
                app=self.app,
            )
        print("Sandbox created successfully.")

        # ─── PRE-FLIGHT VERIFICATION ───
        # Detect working Python executable and verify environment
        # Raises INFRASTRUCTURE_ERROR if issues found (caught by orchestration3)
        print("Running pre-flight verification...")
        self._detect_python()
        self._verify_environment()
        print("Pre-flight verification passed.")

        return self.sandbox

    def ensure_checkpoint(self, checkpoint_name: str) -> str:
        """
        Download checkpoint if not already cached.

        Checkpoints are stored in a Modal Volume for persistence across runs.
        Supports:
        - CHECKPOINT_AUTO_DOWNLOAD: Framework handles download internally (skip this method)
        - CHECKPOINT_DOWNLOAD_CMD: Script-based download (e.g., "bash get_model_params.sh ./model_params")
        - CHECKPOINT_URL: Direct URL download with wget
        - Various formats (.ckpt, .pt, .pth, or no extension)
        """
        # ─── SCRIPT-BASED CHECKPOINT DOWNLOAD ───
        # Use the repo's own checkpoint download script
        checkpoint_dir = f"{WORKING_DIR}/openfold/resources"  # Common location

        # Check if checkpoints already exist
        p = self.sandbox.exec(
            "bash", "-c",
            f"ls {checkpoint_dir}/*.pt 2>/dev/null || ls {checkpoint_dir}/*.ckpt 2>/dev/null || echo MISSING"
        )
        result = p.stdout.read().strip()

        if "MISSING" in result or not result:
            print(f"Running checkpoint download command: {CHECKPOINT_DOWNLOAD_CMD}")
            p = self.sandbox.exec(
                "bash", "-c",
                f"cd {WORKING_DIR} && {CHECKPOINT_DOWNLOAD_CMD}"
            )
            # Stream output
            for line in p.stdout:
                print(line, end="")
            p.wait()
            if p.returncode != 0:
                stderr = p.stderr.read()
                raise RuntimeError(f"Checkpoint download failed: {stderr}")
            print("Checkpoints downloaded successfully.")
        else:
            print(f"Checkpoints already exist in {checkpoint_dir}")

        # Return the checkpoint directory or specific file path
        # The repo's scripts typically handle the exact path
        return checkpoint_dir

    def _upload_file_chunked(self, local_path: Path, remote_path: str, chunk_size: int = 40000) -> None:
        """
        Upload a file to sandbox using chunked base64 transfer.

        Modal has a 64KB ARG_MAX limit, so we split large files into chunks
        and concatenate them on the remote side.

        Args:
            local_path: Path to local file
            remote_path: Destination path in sandbox
            chunk_size: Max bytes per chunk (default 40KB to stay well under 64KB after base64 encoding)
        """
        import base64

        file_bytes = local_path.read_bytes()
        total_size = len(file_bytes)

        if total_size == 0:
            # Empty file - just touch it
            p = self.sandbox.exec("bash", "-c", f"touch {remote_path}")
            p.wait()
            return

        # Use stdin streaming to avoid 64KB ARG_MAX limit for command-line arguments
        file_content = base64.b64encode(file_bytes).decode('ascii')

        # Stream via stdin - works for files of any size
        p = self.sandbox.exec("bash", "-c", f"base64 -d > {remote_path}")
        # Write content to stdin in chunks to avoid memory issues
        stdin_chunk_size = 32768
        for i in range(0, len(file_content), stdin_chunk_size):
            p.stdin.write(file_content[i:i + stdin_chunk_size].encode('ascii'))
        p.stdin.write_eof()
        p.wait()

    def _upload_directory(self, local_dir: Path, remote_dir: str) -> int:
        """
        Upload an entire directory to sandbox recursively.

        Args:
            local_dir: Path to local directory
            remote_dir: Destination directory path in sandbox

        Returns:
            Number of files uploaded
        """
        import base64

        # Create remote directory
        p = self.sandbox.exec("bash", "-c", f"mkdir -p {remote_dir}")
        p.wait()

        uploaded_count = 0
        for local_file in local_dir.rglob("*"):
            if local_file.is_file():
                # Preserve relative path structure
                rel_path = local_file.relative_to(local_dir)
                remote_path = f"{remote_dir}/{rel_path}"

                # Ensure parent directory exists
                remote_parent = str(Path(remote_path).parent)
                p = self.sandbox.exec("bash", "-c", f"mkdir -p {remote_parent}")
                p.wait()

                # Upload file
                self._upload_file_chunked(local_file, remote_path)
                uploaded_count += 1

        return uploaded_count

    def upload_inputs(self, **input_files) -> Dict[str, str]:
        """
        Upload input files/directories to the sandbox.

        Args:
            **input_files: Keyword arguments mapping input names to local paths
                          e.g., protein="examples/protein.pdb", msa_dir="examples/msas/"

        Returns:
            Dict mapping input names to remote paths
        """
        remote_paths = {}

        if "input_fasta" in input_files:
            local_path = Path(input_files["input_fasta"])
            remote_path = "/root/input.fasta".format(**input_files)
            input_type = "file"

            if input_type == "directory":
                # DIRECTORY INPUT: Upload entire directory recursively
                if local_path.is_dir():
                    file_count = self._upload_directory(local_path, remote_path)
                    remote_paths["input_fasta"] = remote_path
                    print(f"  Uploaded directory: {local_path.name}/ ({file_count} files) → {remote_path}")
                else:
                    print(f"  Warning: input_fasta expected directory but got file: {local_path}")
                    # Fallback: treat as single file
                    self._upload_file_chunked(local_path, remote_path)
                    remote_paths["input_fasta"] = remote_path
            else:
                # FILE INPUT: Upload single file
                self._upload_file_chunked(local_path, remote_path)
                remote_paths["input_fasta"] = remote_path
                print(f"  Uploaded: {local_path.name} → {remote_path}")
        if "template_mmcif_dir" in input_files:
            local_path = Path(input_files["template_mmcif_dir"])
            remote_path = "/root/templates/".format(**input_files)
            input_type = "directory"

            if input_type == "directory":
                # DIRECTORY INPUT: Upload entire directory recursively
                if local_path.is_dir():
                    file_count = self._upload_directory(local_path, remote_path)
                    remote_paths["template_mmcif_dir"] = remote_path
                    print(f"  Uploaded directory: {local_path.name}/ ({file_count} files) → {remote_path}")
                else:
                    print(f"  Warning: template_mmcif_dir expected directory but got file: {local_path}")
                    # Fallback: treat as single file
                    self._upload_file_chunked(local_path, remote_path)
                    remote_paths["template_mmcif_dir"] = remote_path
            else:
                # FILE INPUT: Upload single file
                self._upload_file_chunked(local_path, remote_path)
                remote_paths["template_mmcif_dir"] = remote_path
                print(f"  Uploaded: {local_path.name} → {remote_path}")

        return remote_paths

    def run_prediction(
        self,
        remote_inputs: Dict[str, str],
        checkpoint_path: str,
        output_name: str,
        **extra_args
    ) -> PredictionResult:
        """
        Run prediction command and capture result.

        Args:
            remote_inputs: Dict mapping input names to remote paths
            checkpoint_path: Path to checkpoint file in sandbox
            output_name: Name for output file
            **extra_args: Additional arguments to pass to prediction command

        Returns:
            PredictionResult with success status and captured output
        """
        # Build command with all placeholders filled
        cmd = PREDICTION_CMD.format(
            **remote_inputs,
            checkpoint=checkpoint_path,
            output=output_name,
            **extra_args,
        )

        # Substitute hardcoded 'python' with detected python_cmd
        # This handles cases where the Docker image uses conda/micromamba environments
        import re
        if re.match(r'^python3?\s', cmd):
            # Replace 'python' or 'python3' at start with detected python command
            cmd = re.sub(r'^python3?\s', f'{self.python_cmd} ', cmd)
        elif ' python ' in cmd or ' python3 ' in cmd:
            # Handle 'python -m module' style commands embedded in other commands
            cmd = cmd.replace(' python ', f' {self.python_cmd} ')
            cmd = cmd.replace(' python3 ', f' {self.python_cmd} ')

        print(f"\n{'═'*70}")
        print(f"Running prediction...")
        print(f"Command: {cmd}")
        print(f"{'═'*70}\n")

        p = self.sandbox.exec("bash", "-c", cmd, timeout=PREDICTION_TIMEOUT_SECONDS)

        # Stream stdout in real-time
        stdout_lines = []
        for line in p.stdout:
            print(line, end="")
            stdout_lines.append(line)

        # Capture stderr
        stderr = p.stderr.read()
        if stderr:
            print(f"\nSTDERR:\n{stderr}")

        p.wait()

        return PredictionResult(
            success=(p.returncode == 0),
            stdout="".join(stdout_lines),
            stderr=stderr,
            returncode=p.returncode,
        )

    def download_outputs(self, output_name: str, local_dir: Optional[Path] = None) -> List[Path]:
        """
        Download output files from sandbox.

        Args:
            output_name: Base name for output file(s) (e.g., "samples" becomes "samples.sdf")
            local_dir: Local directory to save to (defaults to current dir)

        Returns:
            List of paths to downloaded files

        Note:
            Handles multiple output types. Each output in OUTPUTS has:
            - remote_path: Pattern with {output} placeholder (e.g., "/root/{output}")
            - extension: File extension (e.g., ".sdf")
            - output_type: "file" or "directory" (default: "file")
            - name: Output identifier for error messages
        """
        local_dir = local_dir or Path.cwd()
        downloaded = []
        import base64

        try:
            # Build remote path pattern using the output name and pattern
            remote_pattern = "/root/output/".format(output=output_name)
            extension = ".pdb"
            output_type = "directory"

            if output_type == "directory":
                # DIRECTORY OUTPUT: Download all files from the directory recursively
                print(f"  Downloading directory output: predictions from {remote_pattern}")

                # Create local subdirectory for this output
                output_local_dir = local_dir / "predictions"
                output_local_dir.mkdir(parents=True, exist_ok=True)

                # Find all files in the directory (limit to reasonable count)
                find_cmd = f"find {remote_pattern} -type f 2>/dev/null | head -500"
                find_result = self.sandbox.exec("bash", "-c", find_cmd)
                find_result.wait()
                files = [f.strip() for f in find_result.stdout.read().split('\n') if f.strip()]

                if files:
                    for remote_file in files:
                        # Preserve relative path structure within the output directory
                        try:
                            rel_path = Path(remote_file).relative_to(remote_pattern)
                        except ValueError:
                            rel_path = Path(remote_file).name

                        local_path = output_local_dir / rel_path
                        local_path.parent.mkdir(parents=True, exist_ok=True)

                        p = self.sandbox.exec("bash", "-c", f"base64 {remote_file}")
                        encoded_data = p.stdout.read().strip()
                        p.wait()
                        if encoded_data:
                            data = base64.b64decode(encoded_data)
                            local_path.write_bytes(data)
                            downloaded.append(local_path)
                    print(f"  Downloaded {len(files)} files from predictions directory")
                else:
                    print(f"  Warning: No files found in directory: {remote_pattern}")
            else:
                # FILE OUTPUT: Use glob pattern to find matching files
                # Also try fallback patterns using WORKING_DIR (not hardcoded /root/)
                patterns_to_try = [
                    remote_pattern,
                    f"{WORKING_DIR}/{output_name}{extension}",
                    f"{WORKING_DIR}/outputs/{output_name}{extension}",
                    f"{WORKING_DIR}/outputs/*/{output_name}*{extension}",
                ]

                found_files = []
                for pattern in patterns_to_try:
                    find_cmd = f"ls -1 {pattern} 2>/dev/null || true"
                    find_result = self.sandbox.exec("bash", "-c", find_cmd)
                    find_result.wait()
                    files = [f.strip() for f in find_result.stdout.read().split('\n') if f.strip()]
                    if files:
                        found_files.extend(files)
                        break  # Use first pattern that finds files

                if found_files:
                    for remote_file in found_files:
                        # Create unique local filename for each file
                        file_basename = Path(remote_file).name
                        local_path = local_dir / file_basename

                        p = self.sandbox.exec("bash", "-c", f"base64 {remote_file}")
                        encoded_data = p.stdout.read().strip()
                        p.wait()
                        if encoded_data:
                            data = base64.b64decode(encoded_data)
                            local_path.write_bytes(data)
                            downloaded.append(local_path)
                            print(f"  Downloaded predictions: {remote_file} → {local_path}")
                else:
                    print(f"  Warning: Could not find predictions at pattern: {remote_pattern}")
        except Exception as e:
            print(f"  Warning: Could not download predictions (Predicted protein structure files): {e}")
        try:
            # Build remote path pattern using the output name and pattern
            remote_pattern = "/root/output/alignments/".format(output=output_name)
            extension = ".a3m"
            output_type = "directory"

            if output_type == "directory":
                # DIRECTORY OUTPUT: Download all files from the directory recursively
                print(f"  Downloading directory output: alignments from {remote_pattern}")

                # Create local subdirectory for this output
                output_local_dir = local_dir / "alignments"
                output_local_dir.mkdir(parents=True, exist_ok=True)

                # Find all files in the directory (limit to reasonable count)
                find_cmd = f"find {remote_pattern} -type f 2>/dev/null | head -500"
                find_result = self.sandbox.exec("bash", "-c", find_cmd)
                find_result.wait()
                files = [f.strip() for f in find_result.stdout.read().split('\n') if f.strip()]

                if files:
                    for remote_file in files:
                        # Preserve relative path structure within the output directory
                        try:
                            rel_path = Path(remote_file).relative_to(remote_pattern)
                        except ValueError:
                            rel_path = Path(remote_file).name

                        local_path = output_local_dir / rel_path
                        local_path.parent.mkdir(parents=True, exist_ok=True)

                        p = self.sandbox.exec("bash", "-c", f"base64 {remote_file}")
                        encoded_data = p.stdout.read().strip()
                        p.wait()
                        if encoded_data:
                            data = base64.b64decode(encoded_data)
                            local_path.write_bytes(data)
                            downloaded.append(local_path)
                    print(f"  Downloaded {len(files)} files from alignments directory")
                else:
                    print(f"  Warning: No files found in directory: {remote_pattern}")
            else:
                # FILE OUTPUT: Use glob pattern to find matching files
                # Also try fallback patterns using WORKING_DIR (not hardcoded /root/)
                patterns_to_try = [
                    remote_pattern,
                    f"{WORKING_DIR}/{output_name}{extension}",
                    f"{WORKING_DIR}/outputs/{output_name}{extension}",
                    f"{WORKING_DIR}/outputs/*/{output_name}*{extension}",
                ]

                found_files = []
                for pattern in patterns_to_try:
                    find_cmd = f"ls -1 {pattern} 2>/dev/null || true"
                    find_result = self.sandbox.exec("bash", "-c", find_cmd)
                    find_result.wait()
                    files = [f.strip() for f in find_result.stdout.read().split('\n') if f.strip()]
                    if files:
                        found_files.extend(files)
                        break  # Use first pattern that finds files

                if found_files:
                    for remote_file in found_files:
                        # Create unique local filename for each file
                        file_basename = Path(remote_file).name
                        local_path = local_dir / file_basename

                        p = self.sandbox.exec("bash", "-c", f"base64 {remote_file}")
                        encoded_data = p.stdout.read().strip()
                        p.wait()
                        if encoded_data:
                            data = base64.b64decode(encoded_data)
                            local_path.write_bytes(data)
                            downloaded.append(local_path)
                            print(f"  Downloaded alignments: {remote_file} → {local_path}")
                else:
                    print(f"  Warning: Could not find alignments at pattern: {remote_pattern}")
        except Exception as e:
            print(f"  Warning: Could not download alignments (Multiple sequence alignment files): {e}")
        try:
            # Build remote path pattern using the output name and pattern
            remote_pattern = "/root/output/timings/".format(output=output_name)
            extension = ".json"
            output_type = "directory"

            if output_type == "directory":
                # DIRECTORY OUTPUT: Download all files from the directory recursively
                print(f"  Downloading directory output: timings from {remote_pattern}")

                # Create local subdirectory for this output
                output_local_dir = local_dir / "timings"
                output_local_dir.mkdir(parents=True, exist_ok=True)

                # Find all files in the directory (limit to reasonable count)
                find_cmd = f"find {remote_pattern} -type f 2>/dev/null | head -500"
                find_result = self.sandbox.exec("bash", "-c", find_cmd)
                find_result.wait()
                files = [f.strip() for f in find_result.stdout.read().split('\n') if f.strip()]

                if files:
                    for remote_file in files:
                        # Preserve relative path structure within the output directory
                        try:
                            rel_path = Path(remote_file).relative_to(remote_pattern)
                        except ValueError:
                            rel_path = Path(remote_file).name

                        local_path = output_local_dir / rel_path
                        local_path.parent.mkdir(parents=True, exist_ok=True)

                        p = self.sandbox.exec("bash", "-c", f"base64 {remote_file}")
                        encoded_data = p.stdout.read().strip()
                        p.wait()
                        if encoded_data:
                            data = base64.b64decode(encoded_data)
                            local_path.write_bytes(data)
                            downloaded.append(local_path)
                    print(f"  Downloaded {len(files)} files from timings directory")
                else:
                    print(f"  Warning: No files found in directory: {remote_pattern}")
            else:
                # FILE OUTPUT: Use glob pattern to find matching files
                # Also try fallback patterns using WORKING_DIR (not hardcoded /root/)
                patterns_to_try = [
                    remote_pattern,
                    f"{WORKING_DIR}/{output_name}{extension}",
                    f"{WORKING_DIR}/outputs/{output_name}{extension}",
                    f"{WORKING_DIR}/outputs/*/{output_name}*{extension}",
                ]

                found_files = []
                for pattern in patterns_to_try:
                    find_cmd = f"ls -1 {pattern} 2>/dev/null || true"
                    find_result = self.sandbox.exec("bash", "-c", find_cmd)
                    find_result.wait()
                    files = [f.strip() for f in find_result.stdout.read().split('\n') if f.strip()]
                    if files:
                        found_files.extend(files)
                        break  # Use first pattern that finds files

                if found_files:
                    for remote_file in found_files:
                        # Create unique local filename for each file
                        file_basename = Path(remote_file).name
                        local_path = local_dir / file_basename

                        p = self.sandbox.exec("bash", "-c", f"base64 {remote_file}")
                        encoded_data = p.stdout.read().strip()
                        p.wait()
                        if encoded_data:
                            data = base64.b64decode(encoded_data)
                            local_path.write_bytes(data)
                            downloaded.append(local_path)
                            print(f"  Downloaded timings: {remote_file} → {local_path}")
                else:
                    print(f"  Warning: Could not find timings at pattern: {remote_pattern}")
        except Exception as e:
            print(f"  Warning: Could not download timings (Timing information): {e}")

        return downloaded

    def sync_local_changes(self):
        """
        Sync local code changes into the running sandbox.

        This is the KEY feature that enables iterative debugging:
        - Claude edits local files
        - Changes are synced to running sandbox
        - Prediction re-runs without rebuilding image

        Syncs files matching SYNC_PATTERNS from local repo to sandbox.
        Excludes .venv, __pycache__, .git, and other non-source directories.
        """
        print("Syncing local changes to sandbox...")
        synced_count = 0

        # Directories to always exclude from sync (these are never source code)
        SYNC_EXCLUDE_DIRS = {'.venv', 'venv', '.git', '__pycache__', '.pytest_cache',
                            'node_modules', '.mypy_cache', '.tox', 'dist', 'build',
                            '.eggs', 'site-packages'}

        for pattern in SYNC_PATTERNS:
            for local_path in self.local_dir.glob(pattern):
                if local_path.is_file():
                    rel_path = local_path.relative_to(self.local_dir)

                    # Skip files in excluded directories
                    path_parts = rel_path.parts
                    if any(excluded in path_parts for excluded in SYNC_EXCLUDE_DIRS):
                        continue

                    remote_path = f"{WORKING_DIR}/{rel_path}"

                    try:
                        # Ensure parent directory exists
                        parent = str(Path(remote_path).parent)
                        self.sandbox.exec("bash", "-c", f"mkdir -p {parent}").wait()

                        # Upload file using base64 + stdin streaming
                        # This avoids the 64KB ARG_MAX limit for command-line arguments
                        import base64
                        file_content = base64.b64encode(local_path.read_bytes()).decode('ascii')

                        # Use stdin instead of embedding in command to handle large files
                        p = self.sandbox.exec("bash", "-c", f"base64 -d > {remote_path}")
                        # Write content to stdin in chunks
                        chunk_size = 32768
                        for i in range(0, len(file_content), chunk_size):
                            p.stdin.write(file_content[i:i + chunk_size].encode('ascii'))
                        p.stdin.write_eof()
                        p.wait()
                        print(f"  Synced: {rel_path}")
                        synced_count += 1
                    except Exception as e:
                        print(f"  Failed to sync {rel_path}: {e}")

        print(f"Synced {synced_count} file(s).")

    def terminate(self):
        """Terminate the sandbox and clean up resources."""
        if self.sandbox:
            print("Terminating sandbox...")
            self.sandbox.terminate()
            self.sandbox = None


# ════════════════════════════════════════════════════════════════════════════
# CLAUDE AGENT INTEGRATION - THE CORE DEBUGGING LOGIC
# ════════════════════════════════════════════════════════════════════════════

async def fix_error_with_claude(
    client: ClaudeSDKClient,
    error_stdout: str,
    error_stderr: str,
    attempt: int,
):
    """
    Send error to Claude Agent and let it fix the code.

    This is the CORE of the autonomous debugging loop:
    1. Claude analyzes the error output
    2. Claude edits local files to fix the issue
    3. Changes are synced to sandbox (handled by caller)
    4. Returns status indicating what to do next

    IMPORTANT: Uses ClaudeSDKClient for session continuity - Claude remembers
    previous fix attempts and won't repeat failed fixes.

    Args:
        client: ClaudeSDKClient instance (maintains conversation history)
        error_stdout: Captured stdout from failed prediction
        error_stderr: Captured stderr from failed prediction
        attempt: Current attempt number

    Returns:
        True - Claude made fixes, should retry in same sandbox
        False - Claude cannot fix (CANNOT_FIX), stop trying
        "NEEDS_REBUILD" - Claude added packages, needs image rebuild
    """

    # ─── CLAUDE FIXING PROMPT ───
    # For first attempt, provide full context. For follow-ups, Claude remembers.
    if attempt == 1:
        prompt = f"""You are debugging {REPO_NAME}. The prediction failed with this error:

STDOUT (last 3000 chars):
{error_stdout[-3000:]}

STDERR (last 2000 chars):
{error_stderr[-2000:] if error_stderr else "None"}

CONTEXT:
- Repository: {REPO_NAME}
- Code runs in REMOTE Modal sandbox (not locally)
- Your file edits get synced automatically to the sandbox
- Working directory in sandbox: {WORKING_DIR}

{REPO_NOTES}

COMMON FIXES:
- FileNotFoundError → fix hardcoded paths in code
- ImportError (relative imports) → fix relative imports, fix paths
- Config errors → edit config files
- Type/attribute errors → fix the code

FOR MISSING PACKAGES (ModuleNotFoundError, ImportError: No module named):
1. Add the package to PIP_PACKAGES list in THIS file (fixmyrepo_debugger.py)
2. Say "NEEDS_REBUILD: added <package> to PIP_PACKAGES"
   This tells the orchestrator to rebuild the sandbox image with the new package.

CRITICAL RULES:
1. Do NOT use 'uv add' or 'pip install' - they don't work in remote sandbox
2. NEVER run 'python', 'uv run python', or verification commands - code runs REMOTELY
3. For missing packages: edit PIP_PACKAGES in THIS file, then say NEEDS_REBUILD
4. After making your fix, immediately say "FIXED" or "NEEDS_REBUILD: reason" and stop

Respond with:
- "FIXED" - if you made code changes that should fix the error
- "NEEDS_REBUILD: <reason>" - if you added packages to PIP_PACKAGES (requires image rebuild)
- "CANNOT_FIX: <reason>" - if this is an environment/image issue you cannot fix"""
    else:
        # Follow-up prompt - Claude remembers previous context
        prompt = f"""The previous fix didn't work. Here's the NEW error (attempt {attempt}/{MAX_RETRIES}):

STDOUT (last 3000 chars):
{error_stdout[-3000:]}

STDERR (last 2000 chars):
{error_stderr[-2000:] if error_stderr else "None"}

Remember: You already tried some fixes. Try a DIFFERENT approach this time.
- If same error → your fix didn't address the root cause
- If new error → progress! But something else is broken now
- If ModuleNotFoundError → add to PIP_PACKAGES in THIS file, say NEEDS_REBUILD

Respond with:
- "FIXED" - code changes made
- "NEEDS_REBUILD: <reason>" - added packages to PIP_PACKAGES
- "CANNOT_FIX: <reason>" - impossible to fix"""

    print(f"\n{'═'*70}")
    print(f"Claude is analyzing error (attempt {attempt}/{MAX_RETRIES})...")
    print(f"{'═'*70}\n")

    fixed = False
    gave_up = False
    needs_rebuild = False

    # Use client.query() for session continuity
    await client.query(prompt)

    async for message in client.receive_response():
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text = block.text
                    print(text)

                    if "FIXED" in text.upper():
                        fixed = True
                    if "CANNOT_FIX" in text.upper() or "GIVE UP" in text.upper():
                        gave_up = True
                    if "NEEDS_REBUILD" in text.upper():
                        needs_rebuild = True
                        # Print signal for orchestration3 to detect
                        print(f"\nNEEDS_REBUILD: {text}")

        if isinstance(message, ResultMessage):
            if message.is_error:
                print(f"Claude encountered an error: {message.result}")
                gave_up = True

    # NEEDS_REBUILD means exit immediately so orchestration3 can rebuild
    if needs_rebuild:
        return "NEEDS_REBUILD"  # Special signal for main loop

    if gave_up:
        return False  # Cannot fix

    return True  # Should retry


async def fix_fixmyrepo_debugger_itself(
    error_stdout: str,
    error_stderr: str,
    cwd: str
) -> bool:
    """
    Meta-fix: When CANNOT_FIX occurs, try to fix THIS debugger file.

    This handles environment/image issues by modifying:
    - CONDA_PACKAGES / PIP_PACKAGES lists
    - DOCKER_IMAGE version
    - PATH / environment variables
    - Image build commands

    After fixing, the script will be re-run from scratch to rebuild.

    Uses ClaudeSDKClient for a dedicated meta-fix session.

    Args:
        error_stdout: Captured stdout from failed run
        error_stderr: Captured stderr from failed run
        cwd: Working directory for file operations

    Returns:
        True if Claude fixed this file (should rebuild)
        False if impossible to fix
    """

    image_context = """
The sandbox is built FROM SCRATCH with micromamba + uv:
- Can add/modify CONDA_PACKAGES list
- Can add/modify PIP_PACKAGES list
- Can modify CONDA_CHANNELS
- Can fix PATH, LD_LIBRARY_PATH, or environment variables
- Can modify the image build commands"""

    prompt = f"""The Modal sandbox environment failed with an error that CANNOT be fixed at runtime.
This is likely an environment/image issue that requires modifying THIS debugger file.

ERROR OUTPUT:
STDOUT (last 4000 chars):
{error_stdout[-4000:]}

STDERR (last 2000 chars):
{error_stderr[-2000:] if error_stderr else "None"}

FILE TO FIX: fixmyrepo_debugger.py (in current directory)
{image_context}

COMMON ENVIRONMENT FIXES:
- Missing conda package → add to CONDA_PACKAGES list
- Missing pip package → add to PIP_PACKAGES list
- Library version mismatch (CXXABI, GLIBC) → fix LD_LIBRARY_PATH or add libstdcxx-ng
- Docker image outdated → change DOCKER_IMAGE to newer tag
- micromamba command error → fix the run_commands syntax
- Path issues → fix PATH or MAMBA_ROOT_PREFIX env vars

CUDA/GPU BUILD-TIME ISSUES (VERY IMPORTANT):
- Error mentions CUDA_HOME, nvcc, or "NoneType + str" in setup.py → Package needs CUDA SDK at build time
- CUDA SDK is NOT available during image build, only at runtime
- FIX: REMOVE the problematic package from PIP_PACKAGES (e.g., openfold, flash-attn, apex)
- These packages often have optional features - the main tool may work without them
- Look for pip install lines in environment.yml that reference git repos with setup.py requiring CUDA

WHEN TO REMOVE PACKAGES (not just add):
- Package fails to build due to missing system dependencies that can't be added
- Package is optional (e.g., openfold for ESMFold features in DiffDock)
- Package requires proprietary/unavailable compilers

CRITICAL RULES:
1. ONLY edit fixmyrepo_debugger.py
2. Focus on the ModalSandboxManager._build_image() method
3. Look at CONDA_PACKAGES, PIP_PACKAGES, .env(), .run_commands()
4. Say "FIXED_DEBUGGER" when done, or "IMPOSSIBLE" if truly unfixable

After fixing, the script will be re-run from scratch to rebuild the image."""

    # Use ClaudeSDKClient for meta-fix session (separate from main debug session)
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Edit"],  # Only need to read and edit the debugger file
        disallowed_tools=["Bash", "Write", "Task", "WebFetch", "WebSearch"],  # Focused scope
        permission_mode="acceptEdits",
        cwd=cwd,
        max_turns=10,
    )

    print(f"\n{'═'*70}")
    print("CANNOT_FIX: Claude is attempting to fix fixmyrepo_debugger.py...")
    print(f"{'═'*70}\n")

    fixed = False
    impossible = False

    async with ClaudeSDKClient(options=options) as meta_client:
        await meta_client.query(prompt)

        async for message in meta_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text = block.text
                        print(text)

                        if "FIXED_DEBUGGER" in text.upper():
                            fixed = True
                        if "IMPOSSIBLE" in text.upper():
                            impossible = True

            if isinstance(message, ResultMessage):
                if message.is_error:
                    print(f"Claude encountered an error: {message.result}")

    return fixed and not impossible


def sync_packages_to_result1() -> bool:
    """
    Sync any packages added to PIP_PACKAGES back to result1.json.

    When Claude adds packages for NEEDS_REBUILD, this persists them
    to result1.json so future runs don't need to rediscover them.

    Returns:
        True if result1.json was updated, False otherwise
    """
    import re
    import ast

    debugger_path = Path(__file__).resolve()

    # Find result1.json path: workspace is {project_root}/workspaces/{repo}/
    # result1.json is at {project_root}/fixmyrepo/results/orch1/sbloop_result1_{repo}.json
    workspace_dir = debugger_path.parent
    project_root = workspace_dir.parent.parent
    result1_path = project_root / "fixmyrepo" / "results" / "orch1" / f"sbloop_result1_{REPO_NAME}.json"

    if not result1_path.exists():
        print(f"Warning: result1.json not found at {result1_path}")
        return False

    try:
        # Extract current PIP_PACKAGES from this debugger file
        debugger_content = debugger_path.read_text()
        match = re.search(r'^PIP_PACKAGES\s*=\s*(\[.*?\])', debugger_content, re.MULTILINE | re.DOTALL)
        if not match:
            print("Warning: Could not find PIP_PACKAGES in debugger file")
            return False

        # Parse the list (handle multiline)
        pip_packages_str = match.group(1)
        # Clean up for ast.literal_eval
        pip_packages_str = pip_packages_str.replace('\n', '').replace('  ', ' ')
        current_packages = ast.literal_eval(pip_packages_str)

        # Load result1.json
        result1 = json.loads(result1_path.read_text())
        original_packages = result1.get("pip_packages", [])

        # Find new packages
        original_set = set(original_packages)
        new_packages = [p for p in current_packages if p not in original_set]

        if not new_packages:
            return False

        # Add new packages to result1
        result1["pip_packages"] = current_packages
        result1_path.write_text(json.dumps(result1, indent=2))

        print(f"\n{'─'*70}")
        print(f"✓ Synced {len(new_packages)} new package(s) to result1.json:")
        for pkg in new_packages:
            print(f"  + {pkg}")
        print(f"{'─'*70}\n")

        return True

    except Exception as e:
        print(f"Warning: Failed to sync packages to result1.json: {e}")
        return False


def rerun_from_scratch(args_list: List[str]) -> int:
    """
    Re-run fixmyrepo_debugger.py from scratch after fixing it.

    This creates a fresh subprocess that will:
    1. Rebuild the Modal image with fixes
    2. Start the debugging loop again

    Args:
        args_list: Command line arguments to pass to re-run

    Returns:
        Exit code from the subprocess
    """
    print(f"\n{'═'*70}")
    print("RE-RUNNING FROM SCRATCH (rebuilding image)...")
    print(f"{'═'*70}\n")

    cmd = ["uv", "run", "python", __file__] + args_list
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


# ════════════════════════════════════════════════════════════════════════════
# MAIN AUTONOMOUS DEBUG LOOP
# ════════════════════════════════════════════════════════════════════════════

async def autonomous_debug_loop(
    input_fasta_path: str,
    template_mmcif_dir_path: str,
    output_dir: str = "/root/output/",
    model_device: str = "cuda:0",
    config_preset: str = "model_1",
    jax_param_path: str = None,
    openfold_checkpoint_path: str = None,
    output_name: str = "/root/output/",
) -> Dict[str, Any]:
    """
    Main autonomous debug loop.

    Flow:
    ┌─────────────────────────────────────────────────────────────────┐
    │  1. Create sandbox (keep alive for all attempts)                │
    │  2. Create Claude client (session continuity for all fixes)     │
    │  3. Upload inputs & ensure checkpoint                           │
    │  4. Run prediction                                              │
    │     ├─ SUCCESS → Download output, return                        │
    │     └─ FAILURE → Send to Claude                                 │
    │         ├─ FIXED → Sync changes, goto 4                         │
    │         └─ CANNOT_FIX → Return for meta-fix                     │
    │  5. Repeat until success or max retries                         │
    └─────────────────────────────────────────────────────────────────┘

    IMPORTANT: Uses ClaudeSDKClient for session continuity. Claude remembers
    all previous fix attempts, making debugging more efficient.

    Returns:
        Dict with:
        - success: bool
        - attempts: int
        - needs_rebuild: bool (if CANNOT_FIX occurred)
        - last_stdout/last_stderr: str (if failed)
    """

    manager = ModalSandboxManager()
    cwd = str(Path(__file__).parent.resolve())

    # Create outputs directory for downloaded results
    outputs_dir = Path(cwd) / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Configure Claude client for the debugging session
    # Session continuity: Claude remembers previous fix attempts
    client_options = ClaudeAgentOptions(
        allowed_tools=["Read", "Edit", "Write", "Glob", "Grep", "Bash"],
        disallowed_tools=["Task", "WebFetch", "WebSearch"],  # Focus on local fixes
        permission_mode="acceptEdits",
        cwd=cwd,
        max_turns=15,
    )

    try:
        # ─── SETUP ───
        manager.create()  # Uses SANDBOX_TIMEOUT_MINUTES from config
        checkpoint_path = manager.ensure_checkpoint("default")  # OpenFold uses script-based download

        print("\nUploading input files...")
        remote_inputs = manager.upload_inputs(
            input_fasta=input_fasta_path,
            template_mmcif_dir=template_mmcif_dir_path,
        )

        attempt = 0

        # ─── MAIN LOOP WITH CLAUDE SESSION ───
        # Using ClaudeSDKClient maintains conversation history across fix attempts
        async with ClaudeSDKClient(options=client_options) as claude_client:
            while attempt < MAX_RETRIES:
                attempt += 1

                result = manager.run_prediction(
                    remote_inputs=remote_inputs,
                    checkpoint_path=checkpoint_path,
                    output_name=output_name,
                    output_dir=output_dir,
                    model_device=model_device,
                    config_preset=config_preset,
                    jax_param_path=jax_param_path or "",
                    openfold_checkpoint_path=openfold_checkpoint_path or "",
                )

                if result.success:
                    print(f"\n{'═'*70}")
                    print(f"✓ SUCCESS on attempt {attempt}!")
                    print(f"{'═'*70}\n")

                    # Download outputs to fixmyrepo/outputs/ directory
                    downloaded = manager.download_outputs(output_name, local_dir=outputs_dir)
                    return {
                        "success": True,
                        "attempts": attempt,
                        "outputs": [str(p) for p in downloaded],
                    }

                # ─── ERROR: Let Claude fix it ───
                print(f"\n{'═'*70}")
                print(f"✗ FAILED (attempt {attempt}/{MAX_RETRIES})")
                print(f"{'═'*70}")

                if attempt >= MAX_RETRIES:
                    print("Max retries reached.")
                    break

                # Pass the client for session continuity
                fix_result = await fix_error_with_claude(
                    claude_client, result.stdout, result.stderr, attempt
                )

                if fix_result == "NEEDS_REBUILD":
                    print("\n" + "═"*70)
                    print("NEEDS_REBUILD: Claude added packages that require image rebuild.")
                    print("Exiting so orchestration3 can rebuild the sandbox image.")
                    print("═"*70 + "\n")

                    # Return special status to trigger rebuild in orchestration3
                    return {
                        "success": False,
                        "attempts": attempt,
                        "needs_rebuild": True,
                        "last_stdout": result.stdout,
                        "last_stderr": result.stderr,
                    }

                if not fix_result:
                    print("\n" + "═"*70)
                    print("CANNOT_FIX: Claude cannot fix this error in the runtime sandbox.")
                    print("This is likely an environment/image issue.")
                    print("═"*70 + "\n")

                    # Return special status to trigger meta-fix
                    return {
                        "success": False,
                        "attempts": attempt,
                        "needs_rebuild": True,
                        "last_stdout": result.stdout,
                        "last_stderr": result.stderr,
                    }

                # ─── SYNC FIXES AND RETRY ───
                print("\nSyncing local changes to sandbox...")
                manager.sync_local_changes()
                print("Re-running prediction with fixes...\n")

        return {"success": False, "attempts": attempt, "needs_rebuild": False}

    finally:
        manager.terminate()


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=f"Autonomous debugger for {REPO_NAME}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
    uv run python fixmyrepo_debugger.py \
        /path/to/input.fasta \
        /path/to/templates/ \
        --output_dir /path/to/output/ \
        --model_device cuda:0 \
        --config_preset model_1

Available checkpoints: (checkpoint download handled by script)
        """
    )

    # Input arguments (generated from INPUTS)
    # IMPORTANT: Handle POSITIONAL vs FLAG arguments differently
    # - Positional args (flag == ""): use the NAME, no required= (positional is always required)
    # - Flag args (flag == "--name"): use the FLAG with required= parameter
    # POSITIONAL argument: input_fasta
    parser.add_argument(
        "input_fasta",
        help="Path to input FASTA file"
    )
    # POSITIONAL argument: template_mmcif_dir
    parser.add_argument(
        "template_mmcif_dir",
        help="Path to directory containing template mmCIF structures"
    )
    # FLAG argument: output_dir
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        required=True,
        help="Output directory for predictions"
    )
    # FLAG argument: model_device
    parser.add_argument(
        "--model_device",
        dest="model_device",
        default="cuda:0",
        help="Device to run model on (cuda:0 or cpu)"
    )
    # FLAG argument: config_preset
    parser.add_argument(
        "--config_preset",
        dest="config_preset",
        default="model_1",
        help="Config preset (model_1, model_2, etc.)"
    )
    # FLAG argument: jax_param_path
    parser.add_argument(
        "--jax_param_path",
        dest="jax_param_path",
        default=None,
        help="Path to JAX parameters"
    )
    # FLAG argument: openfold_checkpoint_path
    parser.add_argument(
        "--openfold_checkpoint_path",
        dest="openfold_checkpoint_path",
        default=None,
        help="Path to OpenFold checkpoint"
    )

    # Output arguments
    parser.add_argument("--output", default="/root/output/", help="Output filename")

    # Checkpoint selection
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=f"Model checkpoint (default: script-based download)"
    )

    # Debug loop configuration
    parser.add_argument("--max_retries", type=int, default=10, help="Max fix attempts per run")
    parser.add_argument("--max_rebuilds", type=int, default=3, help="Max image rebuilds for CANNOT_FIX")
    parser.add_argument("--_rebuild_count", type=int, default=0, help=argparse.SUPPRESS)

    args = parser.parse_args()

    global MAX_RETRIES
    MAX_RETRIES = args.max_retries

    # ─── RUN DEBUG LOOP ───
    result = asyncio.run(autonomous_debug_loop(
        input_fasta_path=args.input_fasta,
        template_mmcif_dir_path=args.template_mmcif_dir,
        output_dir=args.output_dir,
        model_device=args.model_device,
        config_preset=args.config_preset,
        jax_param_path=args.jax_param_path,
        openfold_checkpoint_path=args.openfold_checkpoint_path,
        output_name=args.output,
    ))

    if result["success"]:
        # Sync any packages that were added during debugging to result1.json
        sync_packages_to_result1()

        print(f"\n✓ Completed successfully in {result['attempts']} attempt(s).")
        print(f"Outputs: {', '.join(result.get('outputs', []))}")
        sys.exit(0)

    # ─── HANDLE NEEDS_REBUILD: Rebuild image with new packages ───
    if result.get("needs_rebuild", False):
        rebuild_count = args._rebuild_count

        if rebuild_count >= args.max_rebuilds:
            print(f"\n{'═'*70}")
            print(f"MAX REBUILDS REACHED ({args.max_rebuilds})")
            print("The environment/image issue could not be resolved.")
            print("Manual intervention required.")
            print(f"{'═'*70}\n")
            sys.exit(1)

        print(f"\nAttempting to fix fixmyrepo_debugger.py (rebuild {rebuild_count + 1}/{args.max_rebuilds})...")

        cwd = str(Path(__file__).parent.resolve())
        fixed = asyncio.run(fix_fixmyrepo_debugger_itself(
            result.get("last_stdout", ""),
            result.get("last_stderr", ""),
            cwd
        ))

        if fixed:
            print("\nfixmyrepo_debugger.py was modified. Re-running from scratch...")

            # Build args list for re-run
            # IMPORTANT: Handle POSITIONAL vs FLAG arguments differently
            # - Positional args: just the value (no flag prefix)
            # - Flag args: --flag value pairs
            rerun_args = []

            # First, add positional args IN ORDER (just values, no flags)
            rerun_args.append(args.input_fasta)
            rerun_args.append(args.template_mmcif_dir)

            # Then, add flag args (--flag value pairs)
            rerun_args.extend(["--output_dir", args.output_dir])
            rerun_args.extend(["--model_device", args.model_device])
            rerun_args.extend(["--config_preset", args.config_preset])
            if args.jax_param_path:
                rerun_args.extend(["--jax_param_path", args.jax_param_path])
            if args.openfold_checkpoint_path:
                rerun_args.extend(["--openfold_checkpoint_path", args.openfold_checkpoint_path])

            # Add standard args
            rerun_args.extend([
                "--output", args.output,
                "--max_retries", str(args.max_retries),
                "--max_rebuilds", str(args.max_rebuilds),
                "--_rebuild_count", str(rebuild_count + 1),
            ])

            exit_code = rerun_from_scratch(rerun_args)
            sys.exit(exit_code)
        else:
            print("\nClaude could not fix fixmyrepo_debugger.py. Manual intervention required.")
            sys.exit(1)

    print(f"\n✗ Failed after {result['attempts']} attempt(s).")
    sys.exit(1)


if __name__ == "__main__":
    main()
