#!/bin/bash
# Compact status of an editor run: passes, latest notes, newest workspace files. Read-only.
RUN=${1:-opus55_r1}
P=~/Documents/ai_ethics_research/banking77/wsp-banking77
R=$P/runs/editor/$RUN
W=~/wsp-editor-runs/$RUN
echo "== ledger ($RUN)"
[ -f $R/ledger.jsonl ] && python3 - "$R/ledger.jsonl" <<'EOF'
import json, sys
for l in open(sys.argv[1]):
    e = json.loads(l)
    print(f"  {e['time'][11:16]} #{e.get('n','-')} {e['event']} {e.get('type','')} {e.get('model','')} "
          f"inputs={e.get('inputs','')} tok={e.get('tokens','')} edits={e.get('edits','')} {e.get('request','')}")
EOF
echo "== gpu job running: $(pgrep -f 'wsp_cli.py forward' >/dev/null && echo yes || echo no)"
echo "== edits: $(ls $W/edits 2>/dev/null | tr '\n' ' ')"
echo "== newest files:"; ls -t $W/work $W/acts $W/requests 2>/dev/null | grep -v ':$' | head -6 | sed 's/^/  /'
echo "== acts summaries:"; for f in $(ls -t $W/acts/*.json 2>/dev/null | grep -v meta | head -4); do python3 -c "import json;print('  ',\"$(basename $f)\", json.load(open('$f'))['summary'])"; done
echo "== notes.md (tail):"; tail -${2:-15} $W/notes.md | sed 's/^/  /'
