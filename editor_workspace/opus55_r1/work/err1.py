# Hypothesis: step>=1 read-out errors occur almost only at branching prefixes (true class decisions, e.g. card_ -> payment/arrival),
# not at deterministic spelling steps; quantify per-branch accuracy for the L28 mlp_act read-out and the linear L31 mlp_in read-out.
import sys, pickle, collections, numpy as np
sys.path.insert(0, 'work')
from common import *
m, z, si, step, ycls, tgt = records('p03_practice_L31')
seq_fold = np.load('work/p03_seqfold.npy'); val = (seq_fold == 0)[si]
pref = collections.defaultdict(set)
for l in LABS:
    ids = l['token_ids']
    for j in range(len(ids)): pref[tuple(ids[:j])].add(ids[j])
prefix_of = [tuple(LABS[c]['token_ids'][:k]) for c, k in zip(ycls, step)]
branch = np.array([len(pref[p]) > 1 for p in prefix_of])
gtj = np.array([T2J[t] for t in tgt])
sc = pickle.load(open('work/val_scores_sweep.pkl', 'rb'))
S = sc[(28, 'w4', 1e-4)]
ok = S.argmax(1) == gtj[val]
sv = step[val]; bv = branch[val]
for nm, sel in [('step>=1 branching', (sv >= 1) & bv), ('step>=1 deterministic', (sv >= 1) & ~bv)]:
    print(f'L28 read-out {nm}: n={sel.sum()} acc {ok[sel].mean():.4f} errors {(~ok[sel]).sum()}')
# restrict argmax to legal continuations of the gold prefix (what a prefix-aware decoder would do)
legal = np.zeros((val.sum(), len(TOK)), bool)
for i, p in enumerate(np.array(prefix_of, dtype=object)[val]):
    for t in pref[p]: legal[i, T2J[t]] = True
ok_legal = np.where(legal, S, -np.inf).argmax(1) == gtj[val]
print('L28 with legal-continuation restriction: branching acc %.4f deterministic acc %.4f' % (ok_legal[(sv>=1)&bv].mean(), ok_legal[(sv>=1)&~bv].mean()))
nb = collections.Counter(); eb = collections.Counter()
for i in np.where((sv >= 1) & bv)[0]:
    key = ''.join(LABS[ycls[val][i]]['pieces'][:sv[i]]); nb[key] += 1; eb[key] += (not ok[i])
print('per branch prefix (n, errors):', {k: (nb[k], eb[k]) for k in nb})
