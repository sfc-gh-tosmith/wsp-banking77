TASK. You are given the weights of a language model, Llama 3.1 8B Instruct (32 transformer
layers, hidden size 4096, MLP size 14336, vocabulary 128,256; it runs in 4-bit, and any matrix you
edit runs in bf16). Your job is to make it CLASSIFY BANKING CUSTOMER MESSAGES into one of 77 intents
by EDITING ITS WEIGHTS DIRECTLY. Right now it cannot: asked to classify, it writes prose and gets
0% right.

THE TASK FORMAT (fixed; you cannot change it). The model gets the Llama 3.1 chat template with
this user message:

    Classify this banking customer message into the correct intent.

    Message: {the customer's message}

    Intent:

and replies with greedy decoding. The reply must be exactly the intent label (for example
card_arrival or Refund_not_showing_up), then end the turn (the <|eot_id|> token, id 128009).
data/prompt_format.json shows the exact tokens. The 77 labels are in data/labels.json, with their
token ids.

HOW IT WILL BE MEASURED. After you finish, the edited model classifies a hidden test set of 3,080
customer messages that you never see (they are not in your files and not reachable from here).
A reply counts if, lowercased with surrounding whitespace and all spaces removed, it equals the
label lowercased. The reference to beat: the same base model fine-tuned with gradients
(parameter-efficient fine-tuning, 3 epochs on the same labeled data) scores 91.1% on this test set.
Two more things are measured and
both count:
 (1) retention: ordinary general-knowledge questions (plain chat template, no classification
     instruction) that the unedited model answers correctly must still be answered correctly;
 (2) bleed: replies to unrelated prompts must not contain intent labels.
High accuracy that damages (1) or (2) is not a success.

WHAT COUNTS. Edits that come from understanding the network: find where and how the model
represents what a customer is asking about and how it produces label text, and write the
classification there with changes you can justify with evidence. Out of scope, and refused by
save_edit: the final norm (model.norm.weight), the input embeddings (model.embed_tokens.weight),
the output layer (lm_head.weight), and any edit that only multiplies a whole tensor by a constant.
The architecture, tokenizer and prompt stay exactly as they are.

WHAT YOU HAVE. Work only inside this folder.
1. A weights kernel: run your numpy code with ./kpython your_script.py (numpy, scipy and
   safetensors only; there is no PyTorch, MLX or autograd, and the model cannot run here).
       import wsp_kernel as K
       K.list_tensors("layers.31")     # (name, shape, dtype)
       W = K.load(name)                # ORIGINAL weights, float32, HF layout [out, in]
       W = K.load_current(name)        # your saved edit if any, else the original
       K.save_edit(name, W_new)        # the complete new matrix (saved to edits/)
       K.list_edits(); K.remove_edit(name); K.changed_fraction(); K.config()
   lm_head.weight, model.norm.weight and model.embed_tokens.weight can be loaded (read-only) for
   analysis. Keep a script's memory under about 6 GB: the laptop has 16 GB and the model runner
   needs the rest. Variables do not persist between scripts; save intermediate arrays to work/.
2. Your materials:
   - data/practice.jsonl: 9,233 labeled messages {"text", "label", "label_id"}.
   - data/dev.jsonl: 770 labeled messages (10 per intent), kept apart for honest checks. Do not
     build edits from dev; use it to measure.
   - data/labels.json: the 77 labels, each with its token ids (label tokens + 128009).
   - data/prompt_format.json: the full prompt tokens, the shared prefix, and how a message's tokens
     follow it.
   Write any other texts you need yourself (for example unrelated prompts to check bleed).
3. A forward-pass tool with a budget of 30 passes in total: ./wsp forward requests/<name>.json
   Each pass is one of:
   (a) GENERATE: the model replies to up to 10,000 prompts with greedy decoding.
       {"type": "generate", "model": "edited" or "original", "source": "practice" | "dev" | "custom",
        "indices": "all" | [list] | {"n": 500, "seed": 0} | "path/to/indices.json",
        "texts": [...] or "texts_file": "work/x.jsonl" (for source "custom"),
        "template": "sft" (the task prompt above) or "chat" (the message as a plain user turn),
        "max_new_tokens": 24, "out": "name"}
       Results: acts/<name>.json (every reply, and accuracy when labels are known).
   (b) CAPTURE (teacher-forced, no generation): the model reads up to 10,000 prompts and records
       internal states.
       {"type": "capture", "model": ..., "source": ..., "indices": ..., "template": ...,
        "answer": "none" | "gold" (append the label tokens + 128009) | "custom" (custom texts as
                  {"text": ..., "completion": ...}),
        "positions": "last" (last prompt token, which predicts the first reply token) |
                     "answer" (each position that predicts a reply token) | "all",
        "layers": [0..31], "kinds": ["resid", "resid_mid", "attn_out", "mlp_in", "mlp_act", "mlp_out"],
        "topk": 0..20, "out": "name"}
       resid = residual stream after layer l; resid_mid = after its attention; mlp_in = input of the
       MLP (after the norm); mlp_act = input of mlp.down_proj (14336-d); mlp_out = MLP output.
       Results: acts/<name>.npz (float16 arrays [records, dim], plus seq_index, pos, target_id,
       logprob of each target token when there is an answer, topk ids/logprobs) and acts/<name>.meta.json.
       Positions count from the end of the shared prompt prefix (data/prompt_format.json).
   Limits per pass: 10,000 prompts, 300,000 tokens, 3 GB of returned arrays. A request that fails
   validation is not charged. Speed on this laptop: about 130 tokens per second. Capturing every
   practice prompt takes about 25 minutes; generating labels for all of dev takes a few minutes.
   Run long passes with a long command timeout (up to 3600000 ms) or in the background and poll.
   Free, no budget: ./wsp status, ./wsp tokenize "text", ./wsp detok 128009 1234
4. When you are done: ./wsp submit (freezes edits/ as your answer), then write final_answer.md.

HOW TO WORK.
1. Look before you write: measure where the information about which intent a message expresses is
   represented, and how the model produces label tokens, using capture passes.
2. Test causally: show that a change at a place changes the reply the way you predicted.
3. Check on data you did not use to build the edit (dev), plus unrelated prompts for bleed and
   general questions for retention.
4. Work efficiently: state the hypothesis of each script in its first comment line; print compact
   summaries, never whole arrays or raw vocabulary dumps; keep notes.md up to date (findings with
   numbers, edits tried and their measured effect, the current best edit) so you can recover if the
   conversation is compacted; keep copies of good edits (for example copy edits/ to best_1/).

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or
optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through
./wsp: do not implement or simulate the network's layers on inputs in the kernel (linear read-outs
of recorded states, such as multiplying a recorded hidden state or a proposed change by lm_head and
the final norm to see its effect on logits, are fine); (c) internet access, downloading data or
models, other Python environments, other models, Snowflake, or files outside this folder.
Allowed: any analysis of the weights and of forward-pass results, closed-form computations from
them (for example solving a linear system for a weight change), choosing a few
scalar settings by comparing forward-pass results, and any direct edit you compute that is in scope.

HOW THE SESSION RUNS. Continue until you are confident in your edit or the 30 passes are used.
Then run ./wsp submit and write final_answer.md: what you changed, the evidence for it, and why you
expect it to generalize to the hidden test set.

(Task adapted from Takehi, Teoh, Zhang & Khattab, "Sparks of Weight-Space Programming", MIT CSAIL,
2026, and HomoHabilis/weight-space-programming Experiment 9; CC BY 4.0.)
