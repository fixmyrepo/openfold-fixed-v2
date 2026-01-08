# OpenFold Prediction Run Report

## Status: NOT_FIXABLE

**Reason**: Critical bug in fixmyrepo tool prevents conda environment setup

## Executive Summary

The prediction command could not be executed due to a shell escaping bug in the `tools/modal_sandbox.py` file (line 126) that causes conda_env_yaml image builds to fail. This bug is in the fixmyrepo tooling itself, not in the OpenFold repository, and cannot be fixed within the scope of this run due to path restrictions.

## What Was Attempted

### Phase 0: Preflight Fixes Applied

Successfully fixed several configuration issues in `modal_config.json`:

1. **Added missing example_inputs**: The config had an empty `example_inputs` object. Added:
   - `fasta_dir`: "examples/monomer/fasta_dir"
   - `template_mmcif_dir`: "tests/test_data/mmcifs/"
   - `output_dir`: "/root/output/"
   - `model_device`: "cuda:0"
   - `config_preset`: "model_1"

2. **Corrected input parameter name**: Changed `input_fasta` → `fasta_dir`
   - The script (`run_pretrained_openfold.py` line 400) expects a directory containing FASTA files, not a single file
   - Updated the inputs array and prediction_cmd accordingly

3. **Updated remote_path**: Changed from `/root/input.fasta` to `/root/fasta_dir`

### Phase 1: Sandbox Creation - FAILED

Attempted to create Modal sandbox with command:
```bash
python -m tools.modal_sandbox create \
    --config /Users/avinier/UpsurgeLabs/open-bio-project1/workspaces/openfold/fixmyrepo/openfold_modal_config.json \
    --repo-path /Users/avinier/UpsurgeLabs/open-bio-project1/workspaces/openfold \
    --gpu A10G --timeout 60
```

**Error**: Image build failed with shell syntax error:
```
/bin/sh: 1: Syntax error: EOF in backquote substitution
Terminating task due to error: failed to run builder command "echo \"if [ -n \\\"\\$1\\\" ]; then micromamba activate \\\"`basename \\\"\\$1\\\"\\`\\\" ; fi\" >> /opt/conda/bin/activate": container exit status: 2
```

## Root Cause Analysis

### The Bug

File: `/Users/avinier/UpsurgeLabs/open-bio-project1/fixmyrepo/tools/modal_sandbox.py`
Line: 126

```python
'echo "if [ -n \\"\\$1\\" ]; then micromamba activate \\"`basename \\"\\$1\\"\\`\\" ; fi" >> /opt/conda/bin/activate',
```

The backtick substitution `\`basename \\"\\$1\\"\`\` is incorrectly escaped for shell execution in Modal's `run_commands()`, causing a syntax error.

**Correct escaping should be**:
```python
r'echo "if [ -n \"\$1\" ]; then micromamba activate \"\$(basename \"\$1\")\"; fi" >> /opt/conda/bin/activate',
```

### Why This Cannot Be Fixed

1. **Path Restrictions**: The SAFETY_GUARDRAILS policy restricts all file operations to within the target repository path (`/Users/avinier/UpsurgeLabs/open-bio-project1/workspaces/openfold`). The bug is in `/Users/avinier/UpsurgeLabs/open-bio-project1/fixmyrepo/tools/modal_sandbox.py`, which is outside this boundary.

2. **No Alternative Setup Methods**:
   - **Docker**: Repository notes state "no pre-built Docker image is published to any registry"
   - **pip_install**: OpenFold requires conda-specific packages (cuda, gcc=12.4, bioconda tools: hmmer, hhsuite, kalign2) that cannot be installed via pip alone
   - **setup_script**: Would still require a base conda environment with CUDA support

3. **Classification**: PRE_SANDBOX error (image build failure) that cannot be resolved through modal_config.json changes

## Repository Edits

None - the issue is in the fixmyrepo tool, not the target repository.

## Config Changes Applied

Modified `/Users/avinier/UpsurgeLabs/open-bio-project1/workspaces/openfold/fixmyrepo/openfold_modal_config.json`:

1. Line 2: Updated `prediction_cmd` to use `{fasta_dir}` instead of `{input_fasta}`
2. Lines 4-10: Changed input name from `input_fasta` to `fasta_dir` and updated description/remote_path
3. Lines 186-192: Populated `example_inputs` with valid test data paths

## Recommendations

To fix this issue, the maintainer of fixmyrepo should:

1. Update `tools/modal_sandbox.py` line 126 to use proper shell escaping for the activate script
2. Consider using raw strings (r'...') for shell commands with complex escaping
3. Replace backtick command substitution with `$(...)` syntax which is more reliable

Alternatively, use the following corrected line:
```python
r'echo "if [ -n \"\$1\" ]; then micromamba activate \"\$(basename \"\$1\")\"; fi" >> /opt/conda/bin/activate',
```

## Validation Warning Note

The preflight validation warned: "MISMATCH: --output_dir is store_true in script but input_type='file' in config"

This warning is **incorrect**. Line 420 of `run_pretrained_openfold.py` shows:
```python
parser.add_argument("--output_dir", type=str, default=os.getcwd(), ...)
```

The argument has `type=str`, not `action="store_true"`. The config is correct.
