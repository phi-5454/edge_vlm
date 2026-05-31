# Teacher Cache VM Runbook

This runbook is for caching SmolVLM-256M teacher outputs on a VM with a CUDA GPU.

## Required Data

For caching on the VM, copy this directory:

- `data/the_cauldron_yes_no_vsr_token1000_img512_parquet/`

It must contain:

- `combined.parquet`
- `clevr.parquet`
- `vqav2.parquet`
- `vsr.parquet`
- `images.parquet`

`images.parquet` contains the 512x512 JPEG bytes. The cache script reads images
from parquet and does not need `data/the_cauldron/` or the JPEG sidecar
directory `data/the_cauldron_yes_no_vsr_token1000_img512/images/`.

Default cache output:

- `artifacts/teacher_cache/smolvlm_yes_no_vsr_token1000_img512.jsonl`

Caching uses the original `teacher_prompt` and the 512x512 padded student image.

## Preflight

If the parquet artifact is not present yet, build it locally from the 512x512
sidecar dataset before copying data to the VM. This build step requires
`data/the_cauldron_yes_no_vsr_token1000_img512/`, but the caching step does not:

```bash
uv run python scripts/build_student_img512_parquet_dataset.py --force
```

Check dataset visibility and planned record count:

```bash
uv run python scripts/cache_smolvlm_yes_no_teacher.py \
  --dry-run \
  --shard-count 1 \
  --shard-index 0
```

Check CUDA visibility on the VM:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If the model is already in the Hugging Face cache, set `LOCAL_FILES_ONLY=1`.
Otherwise the default allows Hugging Face model download.

## Run

Single process over the whole dataset:

```bash
scripts/run_smolvlm_yes_no_teacher_cache_vm.sh
```

The launcher and the manual single-process command both write
`artifacts/teacher_cache/smolvlm_yes_no_vsr_token1000_img512.jsonl`.

Manual equivalent:

```bash
uv run python scripts/cache_smolvlm_yes_no_teacher.py \
  --dataset data/the_cauldron_yes_no_vsr_token1000_img512_parquet \
  --output artifacts/teacher_cache/smolvlm_yes_no_vsr_token1000_img512.jsonl \
  --device cuda \
  --torch-dtype float16 \
  --batch-size 8 \
  --decode-workers 8 \
  --cpu-threads 8 \
  --image-processor-backend torchvision \
  --top-k 10 \
  --temperature 1.0 \
  --resume
```

To split work into resumable shards on the same VM:

```bash
SHARD_COUNT=4 SHARD_INDEX=0 scripts/run_smolvlm_yes_no_teacher_cache_vm.sh
SHARD_COUNT=4 SHARD_INDEX=1 scripts/run_smolvlm_yes_no_teacher_cache_vm.sh
SHARD_COUNT=4 SHARD_INDEX=2 scripts/run_smolvlm_yes_no_teacher_cache_vm.sh
SHARD_COUNT=4 SHARD_INDEX=3 scripts/run_smolvlm_yes_no_teacher_cache_vm.sh
```

## Output Contents

Each JSONL record includes:

- hard yes/no label and exact dataset identity
- 512x512 parquet image identity and resize metadata
- teacher prompt, student prompt, prompt hashes, and cache image hash
- teacher input token ids, token strings, and attention mask
- image preprocessing tensor metadata and image-token count
- top-k next-token logits
- standalone yes/no logits, log-likelihoods, yes-minus-no logit, and entropy
- first-token log-likelihoods for yes/no answer variants
- per-record standalone yes/no metrics: prediction, correctness, NLL, L1/L2
  distance to the hard one-hot label, and target probability

The sidecar manifest is written next to the JSONL as `*.manifest.json`. It
includes script arguments, versions, host, CUDA availability, CUDA device count,
and aggregate standalone yes/no accuracy, mean NLL, mean L1, mean L2, mean
squared L2, and mean target probability for records written by that invocation.

## Notes

- `--max-examples N` is only for smoke tests. Omitting it caches the selected
  shard.
- `--force` replaces an existing output. Use `--resume` for interrupted jobs.
- On resume, the script prints `selected`, `completed`, and `remaining` counts.
  The progress bar starts at the completed count.
- `--batch-size 8` batches different prompts and images in one forward pass.
  Increase it while GPU memory permits; lower it if CUDA runs out of memory.
- `--decode-workers 8` parallelizes JPEG decoding. CPU preparation for the next
  batch is prefetched while the GPU handles the current batch.
- `--cpu-threads 8` configures PyTorch CPU worker threads for tensor-based image
  preprocessing. Set it no higher than the VM's available CPU count.
- `--image-processor-backend torchvision` uses the faster tensor-based
  Idefics3 image processor instead of the single-threaded PIL backend.
  Transformers currently substitutes bicubic for one unsupported LANCZOS
  operation, so record this choice and use one backend consistently per cache.
- Answer-variant scoring uses the first next-token logits from the prompt
  forward pass. It does not run continuation steps or rerun the multimodal
  prompt.
