# Hypothesis: label token sequences share prefixes; count tokens/steps and divergence points to size the edit.
import json, collections
labs=json.load(open('data/labels.json'))
pr=[json.loads(l) for l in open('data/practice.jsonl')]
dv=[json.loads(l) for l in open('data/dev.jsonl')]
print('n labels',len(labs),'practice',len(pr),'dev',len(dv))
lens=[len(l['token_ids']) for l in labs]
print('tokens per label incl eot: min',min(lens),'max',max(lens),'mean',sum(lens)/len(lens))
first=collections.Counter(l['token_ids'][0] for l in labs)
print('distinct first tokens',len(first),'shared:',[(l['pieces'][0],c) for l in labs for t,c in [(l['token_ids'][0],first[l['token_ids'][0]])] if c>1][:40])
alltok=set(t for l in labs for t in l['token_ids'])
print('distinct tokens in labels',len(alltok))
# prefix tree: number of (prefix)->next ambiguities
pref=collections.defaultdict(set)
for l in labs:
    ids=l['token_ids']
    for j in range(len(ids)):
        pref[tuple(ids[:j])].add(ids[j])
amb=[(k,v) for k,v in pref.items() if len(v)>1]
print('ambiguous prefixes',len(amb),'of',len(pref))
cnt=collections.Counter(x['label'] for x in pr)
print('practice per class min/max',min(cnt.values()),max(cnt.values()))
import statistics
print('msg chars median',statistics.median(len(x['text']) for x in pr), 'max', max(len(x['text']) for x in pr))
print([l['label'] for l in labs])
