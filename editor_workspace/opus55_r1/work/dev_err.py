# Hypothesis: dev errors are mostly well-formed wrong labels (intent confusions), not malformed text; classify error types.
import sys, json, collections, numpy as np
labs = [l['label'] for l in json.load(open('data/labels.json'))]; L = set(x.lower() for x in labs)
dv = [json.loads(l) for l in open('data/dev.jsonl')]
rows = json.load(open('acts/'+(sys.argv[1] if len(sys.argv)>1 else 'p11_dev_v1')+'.json'))['rows']
norm = lambda s: s.strip().replace(' ', '').lower()
types = collections.Counter(); conf = collections.Counter(); bad = []
first = {l.lower(): l.split('_')[0].lower() for l in labs}
for x, r in zip(dv, rows):
    p = norm(r['pred']); g = x['label'].lower()
    if p == g: types['correct'] += 1
    elif p in L:
        types['wrong label, same first word' if first[p] == first[g] else 'wrong label, different first word'] += 1; conf[(g, p)] += 1
    else: types['not a label'] += 1; bad.append((x['label'], r['pred'][:60]))
print(types); print('top confusions', conf.most_common(12)); print('malformed examples', bad[:10])
acc = {}
for x, r in zip(dv, rows): acc.setdefault(x['label'], []).append(norm(r['pred']) == x['label'].lower())
worst = sorted(acc, key=lambda k: np.mean(acc[k]))[:10]
print('worst classes', [(k, np.mean(acc[k])) for k in worst])
