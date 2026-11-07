# Notes

## Setup facts
- 77 labels, 151 distinct label tokens (incl eot), 43 distinct first tokens, 18 branching prefixes. Mean 5.0 tokens/label incl eot.
- Capture records: one per answer position; step 0 = last prompt token (predicts first label token). pos counts from end of shared prefix.

## Pass log
- P1 p01_layers: practice n=2500 seed0, gold, answer positions, resid L6..31 (500s).

## Findings
- Shrinkage-LDA (equal priors) intent decoding at step0 (5-fold CV over 2500): L22 .854, L26 .866, L29 .868, L31 .868; concat 22+26+31 .874.
  Steps>=1 are .96-.99 (emitted label tokens leak class). Learning curve (L29): 1000->.83, 1500->.86, 2000->.86.
- => Final layer (31) carries intent as well as any layer: plan = edit L31 down_proj (linear readout of mlp_act31) so the
  effect is exactly simulable offline (final resid + delta -> final norm -> lm_head; argmax is invariant to rms).
- P2 p02_gen_general_orig: original replies on work/general.jsonl (412 prompts: 131 retention Qs [orig 131/131 correct], 97 unrelated,
  30 near-miss classification, 154 practice msgs as plain chat). Split fit/check alternating.
- P3 p03_practice_L31: all practice, gold, answer positions, L31 mlp_act + resid (47,757 records, 29 min).
- P4 p04_general_L31: general set teacher-forced on original replies, all positions, L31 mlp_act+resid.
  !! HARNESS QUIRK: topk_ids/logprob are ZERO for 75% of p04 rows -> always compute baseline argmax offline from resid
  (offline argmax == captured top1 on 100% of valid rows).
- P5 p05_practice_L28_30: all practice, answer positions, L28 & L30 mlp_act.

## Findings (cont.)
- Ridge read-out L31 mlp_act -> 151 label-token scores (balanced class weights), held-out 20% practice (seq fold 0):
  intrinsic SEQ .894 (s0 .910), mu~0.3 (evals median 13). Step0 weighted x4: SEQ .902 (s0 .919). Step0-only read-out s0 .925 (mu=1).
  Linear read-out on 4096-d L31 resid only: SEQ .852 (s0 .877) -> nonlinear mlp_act features matter.
- UNGATED down_proj read-out leaks: exact full-vocab check on held-out general replies: lam=1,beta=60 -> 30% positions flip,
  15% flip to label tokens; lam=0 -> 97%. => need multiplicative (SwiGLU) task gate.
- Task gate: linear detector on L31 resid separates in-task reply positions (min .50, p1 .83) from general (p99 -.60);
  only other-domain 'Classify ... intent:' near-misses overlap (max .65).
- Low-variance residual directions: bottom-151 eigendirs have rms ~.08-.16 vs ~.6-1.0 per coordinate, but layer31/unembed read
  them at normal gain (~1.2x random) -> fine as a small private code channel, not invisible.
- PLAN (two-stage): stage1 = L30 down_proj ridge read-out of mlp_act30 -> 151-d code in low-variance dirs (ungated, small);
  stage2 = repurpose ~151 low-importance L31 neurons: gate = task detector (silu ~0 off-task), up = code read-out, down = pinv(E_label) col.
- P6 p06_general_L28_30: general set, all positions, L28/L30 mlp_act (for stage-1 preserve term / leak).

## Findings (cont. 2)
- Freeing L31 neurons (zeroing): 32/64/151 lowest mean-square neurons -> 18/27/52 general greedy flips of 8612 positions (0.2-0.6%).
  Worst-case ranking is worse. In-task flips irrelevant (overridden).
- Gates on reconstructed L31 mlp_in (g2*(resid31-Wd@act31)/rms), bias-free ridge: gate0 task&step0: pos min .82, other-step max -.76,
  general max -.34 except near-miss 'Classify X intent' .86. gate1 task&step>=1: pos min .60, general max .09. (work/gatevec_*.npy)
- Read-outs (held-out practice, intrinsic): L28 joint-w4 mu_rel 1e-4 SEQ .9015; L30 .8955. Step0-only: L28 .9274 (mu_rel 3e-3),
  L30 .9226 (1e-3), ensemble L28+L30 .9302 (3e-4). Later steps .995/step.
- P7 p07_gen_general2_orig: original replies on work/general2.jsonl (336: 166 near-miss incl 130 'Classify this <domain> customer message into the correct intent' in exact task format, 93 unrelated incl banking-topic chat Qs, 77 banking msgs as chat). Merged -> work/genall.jsonl (748).
- P8 LOST: foreground run killed by tool cancel (charged, no output). ALWAYS nohup long passes.
- P9 p09_genall_L31: genall teacher-forced, all positions, L31 mlp_in+mlp_act+resid.
- (P8 was NOT charged; p09 run is pass 8.) P9(=pass 9) p10_practice_L31_mlpin: all practice, answer positions, true L31 mlp_in.

## Findings (cont. 3) -- design pivot to SINGLE-STAGE layer-31
- 4-bit->bf16 switch of L31 down_proj alone (simulated group-64 affine) flips ~1.7% of general greedy tokens (|dr|~3.4): any L31 MLP
  edit pays this floor; freeing 151 neurons (~0.6%) and small codes are below it.
- KEY: 77-way CLASS read-out on plain L31 mlp_in (linear, 4096-d, step0-only fit) = .910 class acc, same as L30 mlp_act read-out (.911).
  Earlier gap (.877) came from target encoding (43 first-token unions), not features. Thresholded class-sum -> first token: .925.
  => drop two-stage; everything in L31 MLP reading unmodified mlp_in (captures of ORIGINAL model are exact inputs of the edit).
- Later steps: errors only at branch points (L28 code read-out: branching .978, deterministic .9997). Linear L31 per-token read-out
  (steps>=1 fit) .9937/step, all-later-right .974; branch-specialised read-outs .983 (card .945->.973, top .928->.987).
- Decomposition intent-mass + spelling: whole ~.898; per-token later read-out + step0 relu-sum: whole .9015 (held-out practice).
- Gates with 166 near-miss negatives (genall): gate0 in-task min .66, held-out general max .02 (near-miss .02); gate1 min .53, general max -.12.
- Reconstructed mlp_in vs true: cos .9957, rel err 9% -> use true mlp_in (P9).

## EDIT v1 (best_1/): single-stage L31 MLP, 188 repurposed neurons (least general output energy), 0.029% weights
- 77 intent neurons: gate=kA*(W0_c - (th+beta) b + beta tau0) [=kA(s_c - th + beta(tau0-1))], up=tau0/kA, down=alphaA*D[first(c)]
- 1 task-on neuron: gate=kA(tau0 - .5 b), up=b/(kA*un), down=U*sum D[first tokens]
- 110 token neurons (steps>=1): gate=kB(tau1 - .2 b), up=W1_k/(kB*.8), down=alphaB*D[tok]
- D = pinv of [label-token unembeddings; 2083 look-alike competitor tokens] (cross-talk max .37). b = bias dir (b.x~1).
- Params: mu 1e-4, wneg .01, th .15, beta 1.5, alphaA 250, U 10, alphaB 250, kA 20, kB 50. Fit on all practice + genall.
- Offline CV (fold0, relu decision rule incl. orig logits): whole .896; 0/19645 held-out general positions changed by new neurons.
- P10 p11_dev_v1: DEV 0.8987 (692/770). Errors: 59 wrong first word, 11 wrong branch, 6 no-fire -> prose, 2 incoherent 'top_up_by_card'.
- P11 p12_genall_v1 (748 general prompts, edited): retention 131/131 correct; bleed 0 (incl. 196 near-miss, 231 banking chat);
  identical first-150-chars: retention 104/131, unrelated 120/190 (diffs from bf16 switch of L31 MLP).
- LDA vs ridge (step0 class): ridge .9107 better than LDA (.90).
- Offline: pair experts (top-2 re-rank) no gain (.9107->.9123). Kernel-ridge concat in31+act30 .9145/.930 vs in31 .9128/.924 (+.5%, noise-level,
  would need 2-stage). Group read-out add +.4% (noise). Later: branch weight x10 + legal gamma .5 -> all-later-right .975->.982.
- EDIT v2 (best_2/): v1 + bw=10, gamma=.5 (W1 targets onehot+.5*legal, up scaled 1/(1+gamma)), U=15. CV whole .9036.
- P12 p13_dev_v2: DEV 0.9078 (699/770). Errors: 62 wrong first word, 7 wrong branch, 2 malformed.
- Write matrix alternatives (cosine competitors, single-shot threshold set): worse relative cross-talk; keep D_write (self 1, label off-diag 0, max .37).
- 5-fold offline decision-rule estimate for v2: whole .9036/.9182/.9053/.8982/.8976 -> mean .9046 (class-balanced ~.902). Dev .9078 agrees.
- P13 p14_genall_v2: retention 131/131, bleed 0 (same as v1).
- Offline (all no/marginal gain): per-branch-point experts all-later-right .978->.979; learned 2nd-layer aggregation .925->.927;
  single-shot competitor sets for D worse (max rel. cross-talk .60 vs .37).
- L28 mlp_act carries more step-0 info (5-fold first-token .933 vs in31 .927; ensemble +.6%), BUT an ungated L28 code is huge
  off-task (|code| general 6x in-task) and constraining it to 0 off-task kills the gain (.921) -> two-stage rejected.
- 5-fold step-0 sweep: wneg .003 mu 2e-4 = +.27% (4/5 folds); fold-0 general risk 0/19645 (max act .019).
- EDIT v3 (best_3/): v2 + wneg=.003, mu=2e-4. P14 p15_dev_v3, P15 p16_genall_v3.
- P14 p15_dev_v3: DEV 0.9117 (702/770); vs v2 fixed 3, broke 0. Errors: 59 wrong first word, 7 wrong branch, 2 malformed.
- P15 p16_genall_v3: retention 131/131, bleed 0, identical-prefix same as v1/v2; other-domain 'Classify ... intent' prompts answered in prose.
- FINAL = v3 (best_3/). Submitted after P15. 15/30 passes used.
