"""Smoke-test the hosted edited Llama through AI_COMPLETE's bring-your-own-model path.

    python byom_smoke.py      (uses .venv-sf)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from submit_merge_job import session  # noqa: E402

SVC = "ML_DEMO.BANKING77_WSP.BANKING77_WSP_SVC"
s = session()
# The private-preview flag ENABLE_SPCS_SERVICE_FUNCTIONS_IN_AISQL is gone in Open Preview (invalid parameter).
rows = s.sql(f"""
    SELECT text, label,
           AI_COMPLETE('{SVC}',
               'Classify this banking customer message into the correct intent.\\n\\nMessage: ' || text || '\\n\\nIntent:',
               {{'temperature': 0, 'max_tokens': 12}}) AS predicted
    FROM ML_DEMO.BANKING77_WSP.B77_DEV
    LIMIT 5""").collect()
for r in rows:
    print(f"{r['LABEL']:<40} -> {r['PREDICTED']!r:<45} | {r['TEXT'][:60]}")
print("general prompt:", s.sql(f"SELECT AI_COMPLETE('{SVC}', 'Say hello in one short sentence.', {{'max_tokens': 30}})").collect()[0][0])
