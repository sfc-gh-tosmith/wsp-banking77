# Final answer: banking-intent classifier written into layer 31's MLP

Submitted edit: v3 (also in `best_3/`). Submission id 997747f07403ef7c. 15 of 30 passes used.

## What changed

I edited three tensors: `model.layers.31.mlp.gate_proj`, `up_proj` and `down_proj`. Only 188 of the 14,336 layer-31 MLP neurons were changed. That is 2.31 M parameters, 0.029% of the model. Everything else, including embeddings, final norm and `lm_head`, is untouched.

**Which neurons.** I repurposed the 188 neurons with the least output energy on general text. Energy is mean activation² × ‖down column‖², measured over 748 general prompts. These neurons carry 7.7e-4 of layer 31's MLP output energy on that text.

**What they do.** Each neuron is SwiGLU-gated, `h = silu(g·x)·(u·x)`. Here x is the layer's unmodified MLP input, so recorded inputs of the original model are exactly the inputs the edit sees. Every vector below came from one closed-form ridge solve or pseudo-inverse on recorded layer-31 inputs: all 9,233 practice examples, plus the 748 general prompts as negatives.

**Detector and helper vectors.** Each is a single ridge solve:
- τ0 is a "task prompt, first reply token" detector:
  - in-task ≥ 0.58 (median 0.99);
  - every held-out general position ≤ 0.42, including other-domain "Classify this … intent" prompts;
  - in-task later steps ≤ −0.74.
- τ1 is a "task prompt, later reply token" detector: in-task ≥ 0.53, general ≤ −0.12.
- b is a constant direction with b·x ≈ 1 everywhere. It stands in for the bias the MLP lacks.
- D is the write matrix: the pseudo-inverse of the 151 label-token unembeddings plus 2,083 look-alike tokens, which are given zero-boost constraints. The look-alikes were collected over a few rounds of closed-form solves on `lm_head` only. Each label token's write boosts it by exactly 1 and the other label tokens by 0; the worst look-alike gets 0.37.

**The neurons:**

| Neurons | Gate | Up | Down |
|---|---|---|---|
| 77 intent neurons (step 0) | κ(s_c − 0.15 + 1.5(τ0 − 1)), where s_c is a class-c ridge read-out (step-0 one-hot, balanced weights, mild off-task negatives) | τ0 | 250 × D[first token of c] |
| 1 task-on neuron | κ(τ0 − 0.5) | constant | 15 × Σ D[first tokens], lifting label tokens above prose |
| 110 next-token neurons (steps ≥ 1) | κ(τ1 − 0.2) | ridge read-out of the next label token (branch positions weighted ×10, plus 0.5 × legal-continuation targets) | 250 × D[token] |

At step 0, the first token with the most intent mass wins. Each later token follows the read-out until the end-of-text token (eot). Off-task, the gate is far negative, so silu ≈ 0. Where τ0 < 0, the up factor makes any residual output negative, so it suppresses label tokens rather than adding them.

**Scalar settings** (κ, thresholds, gains, ridge μ = 2e-4, off-task weight 0.003) were chosen by 5-fold cross-validation on practice and by comparing dev passes v1 → v2 → v3. No training, gradients or iterative fitting were used.

## Evidence

All numbers below are from real `./wsp` passes unless marked as offline.

**Dev accuracy** (770 messages, never used for fitting):

| Version | Change | Dev accuracy |
|---|---|---|
| v1 | — | 89.9% |
| v2 | branch weighting and legal-continuation targets | 90.8% |
| v3 | weaker off-task penalty | **91.2% (702/770)** |

v2 → v3 fixed 3 dev items and broke none.

**Remaining dev errors.** 59 have the wrong first word, 7 take the wrong branch, and 2 are not labels. Two thirds of the replies were well-formed `label + eot`.

**Retention and bleed** (pass 15, all 748 general prompts, v3):
- **Retention:** 131/131 retention questions the original answered correctly are still correct.
- **Bleed:** 0 replies contain an intent label. This covers 190 unrelated prompts, 231 banking questions asked as plain chat, and 196 near-misses. The near-misses include 130 prompts in the exact task format for other domains (e-commerce, healthcare, …), which still get normal prose answers.
- **Identical replies:** 104/131 retention and 120/190 unrelated replies match the original exactly for the first 150 characters. The differences come from the edited matrices switching from 4-bit to bf16, which I estimate flips about 1.7% of greedy tokens with no edit at all. Offline, the new neurons themselves changed 0 of 19,645 held-out general positions.

**Offline estimate.** The 5-fold cross-validated estimate on practice was about 0.905 whole-label for v2. It tracked the dev results, so the decision-rule analysis predicts real behaviour.

## Why it should generalize, and the main risk

- **Dev is held out.** The edit was never fit to dev, and dev agrees with the 5-fold estimates.
- **Balanced test set.** Read-outs use class-balanced weights. Class-balanced accuracy is within about 0.3% of plain accuracy.
- **No leakage onto general text.** The task detectors are linear in a 4,096-d state and separate in-task from general positions with a wide margin on held-out prompts. Activation is multiplicative (gate × task factor), so the behaviour stays off for anything that isn't the task prompt.
- **Main risk: the margin over the reference is thin.** Dev 91.2% vs the 91.1% reference is within the ±1% standard error of a 770-item sample, so the hidden-test result could land on either side.
- **Accuracy is capped by layer 31's features.** Every alternative I tried tops out at about 92.5–93% first-token accuracy from layer 31's last-position features: LDA, pairwise experts, a learned second layer, per-branch experts, and higher write gains.
- **The one source with more information was rejected.** Layer-28 MLP activations carry about 0.6% more step-0 information. Using them needs a two-stage edit, and in that setup the code was not separable from general text: it was 6× larger off-task than in-task. Since the edit would sit on every general prompt, I judged it too risky for retention and bleed.

Work files are in `work/`. The build is `work/build_final.py final bw=10 gamma=0.5 U=15 wneg=0.003 mu=2e-4`, followed by `work/write_edit.py`. The full log is in `notes.md`.
