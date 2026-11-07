# Hypothesis: edit v1 leaves general behaviour intact: retention answers still correct, no intent label strings in any reply, and most
# replies identical to the original's (differences only from the unavoidable 4-bit->bf16 switch of the edited matrices).
import json, re, collections, sys
name = sys.argv[1] if len(sys.argv) > 1 else 'p12_genall_v1'
rows = [json.loads(l) for l in open('work/genall.jsonl')]
new = json.load(open(f'acts/{name}.json'))['rows']
labs = [l['label'] for l in json.load(open('data/labels.json'))]
lab_re = re.compile('|'.join(re.escape(l.lower()) for l in sorted(labs, key=len, reverse=True)))
stats = collections.defaultdict(lambda: collections.Counter()); bleed = []; ret_bad = []
for r, n in zip(rows, new):
    k = r['kind']; o = r['completion']; p = n['pred']
    L = min(len(o), len(p), 150)
    stats[k]['n'] += 1; stats[k]['identical_first150'] += (o[:L] == p[:L])
    hits = lab_re.findall(p.lower().replace(' ', ''))
    hits_o = lab_re.findall(o.lower().replace(' ', ''))
    if hits and not hits_o: stats[k]['bleed'] += 1; bleed.append((k, r['text'][:60], p[:80]))
    if hits_o: stats[k]['label_in_original'] += 1
    if k == 'retention':
        ok_o = any(a in o.lower() for a in r['answers']); ok_n = any(a in p.lower() for a in r['answers'])
        stats[k]['orig_correct'] += ok_o; stats[k]['edit_correct'] += ok_n
        if ok_o and not ok_n: ret_bad.append((r['text'], p[:100]))
for k, c in stats.items(): print(k, dict(c))
print('bleed examples:', bleed[:8]); print('retention lost:', ret_bad[:8])
