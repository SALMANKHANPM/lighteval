# LightEval Repository Changes

This document lists all modifications made to the LightEval codebase and related packages to fix compatibility issues.

## Summary

- **Total files modified**: 4 files
- **LightEval changes**: 2 files
- **vLLM changes**: 1 file  
- **SGLang changes**: 1 file

---

## 1. LightEval Changes

### File: `lighteval/utils/imports.py`

**Location**: `/home/safertek/Projects/lid/.venv/lib/python3.12/site-packages/lighteval/utils/imports.py`

**Line**: ~60-66 (in `is_package_available()` function)

**Change Type**: Added code

**Description**: Relaxed version check to allow newer vLLM versions

**Reason**: LightEval's pyproject.toml pins vllm to `>=0.10.0,<0.10.2`, but vllm 0.15.1 was installed. This caused an ImportError when trying to use VLLMModel.

**Code Added**:
```python
# Allow newer vllm than lighteval's pin (e.g. 0.15.x when pyproject has 0.10.x)
if package.name == "vllm" and installed >= Version("0.10.0"):
    return True
```

**Before**:
```python
else:
    try:
        installed = Version(version(package.name))
    except PackageNotFoundError:
        return False

    # No version constraint → any installed version is OK
    if not package.specifier:
        return True

    return installed in package.specifier
```

**After**:
```python
else:
    try:
        installed = Version(version(package.name))
    except PackageNotFoundError:
        return False

    # No version constraint → any installed version is OK
    if not package.specifier:
        return True

    # Allow newer vllm than lighteval's pin (e.g. 0.15.x when pyproject has 0.10.x)
    if package.name == "vllm" and installed >= Version("0.10.0"):
        return True

    return installed in package.specifier
```

---

### File: `lighteval/main_vllm.py`

**Location**: `/home/safertek/Projects/lid/.venv/lib/python3.12/site-packages/lighteval/main_vllm.py`

#### Change 1: Import Statement

**Line**: ~28

**Change Type**: Added import

**Description**: Added `load_tasks_multilingual` to imports from `lighteval.cli_args`

**Code Added**:
```python
load_tasks_multilingual,
```

**Before**:
```python
from lighteval.cli_args import (
    HELP_PANEL_NAME_4,
    custom_tasks,
    dataset_loading_processes,
    job_id,
    load_responses_from_details_date_id,
    max_samples,
    model_args,
    num_fewshot_seeds,
    output_dir,
    public_run,
    push_to_hub,
    push_to_tensorboard,
    reasoning_tags,
    remove_reasoning_tags,
    results_org,
    results_path_template,
    save_details,
    tasks,
    wandb,
)
```

**After**:
```python
from lighteval.cli_args import (
    HELP_PANEL_NAME_4,
    custom_tasks,
    dataset_loading_processes,
    job_id,
    load_responses_from_details_date_id,
    load_tasks_multilingual,
    max_samples,
    model_args,
    num_fewshot_seeds,
    output_dir,
    public_run,
    push_to_hub,
    push_to_tensorboard,
    reasoning_tags,
    remove_reasoning_tags,
    results_org,
    results_path_template,
    save_details,
    tasks,
    wandb,
)
```

#### Change 2: Function Parameter

**Line**: ~59

**Change Type**: Added parameter

**Description**: Added `load_tasks_multilingual` parameter to `vllm()` function signature

**Code Added**:
```python
load_tasks_multilingual: load_tasks_multilingual.type = load_tasks_multilingual.default,
```

**Before**:
```python
    # === Common parameters ===
    cot_prompt: Annotated[
        Optional[str], Option(help="Use chain of thought prompt for evaluation.", rich_help_panel=HELP_PANEL_NAME_4)
    ] = None,
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
```

**After**:
```python
    # === Common parameters ===
    cot_prompt: Annotated[
        Optional[str], Option(help="Use chain of thought prompt for evaluation.", rich_help_panel=HELP_PANEL_NAME_4)
    ] = None,
    load_tasks_multilingual: load_tasks_multilingual.type = load_tasks_multilingual.default,
    dataset_loading_processes: dataset_loading_processes.type = dataset_loading_processes.default,
    custom_tasks: custom_tasks.type = custom_tasks.default,
```

#### Change 3: PipelineParameters Initialization

**Line**: ~100

**Change Type**: Added parameter

**Description**: Added `load_tasks_multilingual` parameter when creating `PipelineParameters`

**Code Added**:
```python
load_tasks_multilingual=load_tasks_multilingual,
```

**Before**:
```python
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        job_id=job_id,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        cot_prompt=cot_prompt,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
    )
```

**After**:
```python
    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        job_id=job_id,
        load_tasks_multilingual=load_tasks_multilingual,
        dataset_loading_processes=dataset_loading_processes,
        custom_tasks_directory=custom_tasks,
        num_fewshot_seeds=num_fewshot_seeds,
        max_samples=max_samples,
        cot_prompt=cot_prompt,
        load_responses_from_details_date_id=load_responses_from_details_date_id,
        remove_reasoning_tags=remove_reasoning_tags,
        reasoning_tags=reasoning_tags,
    )
```

**Reason**: The `--load-tasks-multilingual` flag was available for `sglang` backend but missing for `vllm` backend, causing inconsistency.

---

## 2. vLLM Changes

### File: `vllm/transformers_utils/tokenizer.py`

**Location**: `/home/safertek/Projects/lid/.venv/lib/python3.12/site-packages/vllm/transformers_utils/tokenizer.py`

**Line**: ~99-106 (in `get_cached_tokenizer()` function)

**Change Type**: Modified code

**Description**: Made `all_special_tokens_extended` attribute access optional with fallback

**Reason**: `GemmaTokenizer` (from transformers) doesn't have the `all_special_tokens_extended` attribute that vLLM expects, causing `AttributeError` when loading models like `google/gemma-3-1b-it`.

**Before**:
```python
    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_all_special_tokens_extended = (
        tokenizer.all_special_tokens_extended)
```

**After**:
```python
    tokenizer_all_special_ids = tokenizer.all_special_ids
    tokenizer_all_special_tokens = tokenizer.all_special_tokens
    tokenizer_all_special_tokens_extended = getattr(
        tokenizer,
        "all_special_tokens_extended",
        tokenizer_all_special_tokens,
    )
```

**Impact**: If `all_special_tokens_extended` is missing, falls back to `all_special_tokens`, allowing tokenizers without this attribute to work with vLLM.

---

## 3. SGLang Changes

### File: `sglang/srt/layers/quantization/awq.py`

**Location**: `/home/safertek/Projects/lid/.venv/lib/python3.12/site-packages/sglang/srt/layers/quantization/awq.py`

**Line**: ~44-60 (import section)

**Change Type**: Modified code

**Description**: Made `fused_marlin_moe` import optional with try/except

**Reason**: The installed `sgl_kernel` package doesn't export `fused_marlin_moe`, causing `ImportError` when SGLang tries to load the Engine. This is likely a version mismatch between sglang and sgl_kernel.

**Before**:
```python
if _is_cuda:
    from sgl_kernel import (
        awq_dequantize,
        awq_marlin_moe_repack,
        awq_marlin_repack,
        fused_marlin_moe,
    )
```

**After**:
```python
if _is_cuda:
    from sgl_kernel import (
        awq_dequantize,
        awq_marlin_moe_repack,
        awq_marlin_repack,
    )
    try:
        from sgl_kernel import fused_marlin_moe
    except ImportError:
        fused_marlin_moe = None  # older sgl_kernel may not export this
```

**Line**: ~767 (usage site)

**Change Type**: Added validation

**Description**: Added check to raise helpful error if `fused_marlin_moe` is None when needed

**Code Added**:
```python
        if fused_marlin_moe is None:
            raise RuntimeError(
                "AWQ Marlin MoE requires fused_marlin_moe from sgl_kernel. "
                "Your sgl_kernel version does not export it. "
                "Upgrade sgl_kernel (pip install -U sgl-kernel) or use a non-MoE AWQ model."
            )
```

**Impact**: Allows SGLang to load even when `fused_marlin_moe` is unavailable, only failing with a clear error message when AWQ Marlin MoE is actually used.

---

## Issues Fixed

1. **vLLM version compatibility**: Fixed ImportError when using vllm 0.15.1 with lighteval that expects vllm 0.10.x
2. **GemmaTokenizer compatibility**: Fixed AttributeError when loading Gemma models with vLLM
3. **SGLang kernel compatibility**: Fixed ImportError when loading SGLang Engine due to missing `fused_marlin_moe`
4. **Missing CLI option**: Added `--load-tasks-multilingual` flag to vllm backend for consistency with sglang

---

## Notes

- All changes are in the virtual environment's site-packages directory
- These patches will be overwritten if packages are reinstalled or upgraded
- For permanent fixes, consider:
  - Upgrading/downgrading packages to compatible versions
  - Submitting patches upstream to the respective projects
  - Using a requirements.txt with pinned compatible versions

---

## Testing

After these changes, the following commands should work:

```bash
# vLLM backend with multilingual tasks
HF_DATASETS_TRUST_REMOTE_CODE=1 lighteval vllm \
  --custom-tasks custom_tasks_indicqa.py \
  "model_name=google/gemma-3-1b-it,max_model_length=4096" \
  "indicqa_tel" \
  --max-samples 5 \
  --load-tasks-multilingual

# SGLang backend (if sgl_kernel issues are resolved)
HF_DATASETS_TRUST_REMOTE_CODE=1 lighteval sglang \
  "model_name=google/gemma-3-1b-it" \
  "indicqa_tel" \
  --max-samples 5 \
  --load-tasks-multilingual
```

---

## Git Patches

Git patches are available in the `patches/` directory:

- `patches/lighteval-imports-vllm-version-fix.patch` - vLLM version compatibility fix
- `patches/lighteval-main-vllm-multilingual-option.patch` - Add multilingual option to vllm CLI
- `patches/vllm-tokenizer-gemma-compat.patch` - GemmaTokenizer compatibility fix
- `patches/sglang-awq-fused-marlin-moe-optional.patch` - Optional fused_marlin_moe import
- `patches/apply-all.patch` - Combined patch with all changes

See `patches/README.md` for instructions on how to apply these patches.

---

**Date**: February 20, 2026  
**Environment**: Python 3.12, Linux 6.17.0-14-generic
