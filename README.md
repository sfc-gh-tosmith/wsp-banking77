# Banking77 via weight-space programming (WSP)

Can an LLM agent replace supervised fine-tuning by **editing a model's weights directly**, using only
forward passes and numpy (no gradients, no optimizer)? This repo applies the idea from MIT CSAIL's
"Sparks of Weight-Space Programming" (Nov 2026) to the Banking77 intent-classification benchmark.

An editor agent (Claude Opus 5.5 in a sandboxed Cortex Code chat) was given Llama 3.1 8B Instruct, a budget of 30
forward passes, and the practice split. It edited 188 repurposed neurons in layer 31's MLP
(2.31M weights, 0.029% of the model). It used 15 passes.

## Results (Banking77 test, 3,080 rows, exact match)

| Approach | Test accuracy | Macro-F1 |
|---|---|---|
| B0: base model, SFT prompt | 0% | - |
| B2: scripted closed-form ridge edit, L31 `down_proj` | 52.1% | 0.60 |
| B1: base model, zero-shot with label list | 58.7% | 0.57 |
| Cortex fine-tune reference (llama3.1-8b, PEFT, 3 epochs) | 91.1% | 0.911 |
| **WSP edit (opus55_r1, v3), hosted in Snowflake** | **91.95%** (CI 91.0-92.9) | **0.922** |

The edit keeps general behavior: retention 40/40, zero bleed into non-banking prompts, 0.6% off-label replies.

Lesson learned: the edit was fit on 4-bit activations. Merging it onto a bf16 base dropped accuracy to about 72%.
Rebuilding the exact dequantized 4-bit base (`hosting/merge_q4base_job.py`) recovered 91.2% dev, with 100% per-row
agreement with the local run.

## Layout

- `harness/`: the sandbox and its tooling.
  - `TASK.md` is the editor's brief.
  - `wsp_cli.py` is the budgeted `./wsp` forward-pass tool, with its ledger and snapshots.
  - `wsp_kernel.py` is a numpy-only weight access kernel.
  - `runner.py` is an MLX model runner with a prefix KV cache.
  - `eval.py` and `score_run.py` do scoring; `audit.py` checks the editor transcript for rule violations.
  - `b2_closed_form.py` is the scripted ridge-regression baseline.
- `hosting/`:
  - `pack_edit.py` packs an edit into a bit-exact ~2.3 MB `.npz` (`packed/`).
  - The merge jobs and `submit_merge_job.py` run the merge as a Snowflake ML Job.
  - `create_service.py` creates a vLLM service on SPCS.
  - `byom_pipeline.sql` and `run_byom_pipeline.py` classify messages in SQL via `AI_COMPLETE` on the hosted model.
- `data/`: Banking77 train, plus the dev (10 per intent) and practice splits. The test split is not stored locally.
  It lives only in Snowflake, so the editor can't see it.
- `runs/`: evaluation outputs and the editor run's ledger and audit.
- `baselines/`: the B2 sweep summary.

## Running it

Weights (`models/`) are not included. Download them from Hugging Face:
- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` goes to `models/llama31-8b-instruct-4bit`.
- `unsloth/Meta-Llama-3.1-8B-Instruct` goes to `models/llama31-8b-instruct-bf16`.

Snowflake access uses a named connection from your local `connections.toml`. Set `WSP_SF_CONNECTION`
(default `default`). Objects live in `ML_DEMO.BANKING77_WSP` (see `harness/common.py`). Hosting steps start
GPU/CPU compute pools, so suspend them when you are done.

## Licenses and attribution

- Banking77: PolyAI, CC-BY-4.0 (`data/README_banking77.md`). Casanueva et al., 2020, arXiv:2003.04807.
- Built with Llama. The packed edits in `hosting/packed/` contain modified Llama 3.1 weights and are subject to the
  [Llama 3.1 Community License](https://www.llama.com/llama3_1/license/).
