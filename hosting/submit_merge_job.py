"""Host an edit in Snowflake (run only after explicit approval: it starts compute).

    python submit_merge_job.py --packed <packed.npz> --version V_<RUN> [--setup]

--setup creates (idempotent) the stage, HF network rule + external access integration and a CPU pool
for the merge job, all in ML_DEMO.BANKING77_WSP. Then: PUT the packed edit (a few MB), submit the ML Job that
builds the edited bf16 model and logs it to the Model Registry. Service creation is separate
(create_service.py), as the registry skill requires.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import SF_CONNECTION, SF_DB, SF_SCHEMA, SF_WH  # noqa: E402

MODEL_NAME = "LLAMA_3_1_8B_BANKING77_WSP"
STAGE = f"{SF_DB}.{SF_SCHEMA}.WSP_EDITS"
JOB_STAGE = f"{SF_DB}.{SF_SCHEMA}.WSP_JOB_PAYLOADS"
EAI = "BANKING77_WSP_HF_EAI"
CPU_POOL = "BANKING77_WSP_CPU_POOL"
SETUP_SQL = [
    f"CREATE STAGE IF NOT EXISTS {STAGE} ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
    f"CREATE STAGE IF NOT EXISTS {JOB_STAGE} ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE')",
    f"""CREATE NETWORK RULE IF NOT EXISTS {SF_DB}.{SF_SCHEMA}.BANKING77_WSP_HF_RULE TYPE = HOST_PORT MODE = EGRESS
        VALUE_LIST = ('huggingface.co', '*.huggingface.co', '*.hf.co', '*.xethub.hf.co', '*.cdn.hf.co',
                      '*.aws.cdn.hf.co', '*.gcp.cdn.hf.co')""",
    f"""CREATE EXTERNAL ACCESS INTEGRATION IF NOT EXISTS {EAI}
        ALLOWED_NETWORK_RULES = ({SF_DB}.{SF_SCHEMA}.BANKING77_WSP_HF_RULE) ENABLED = TRUE
        COMMENT = 'Banking77 WSP: download base Llama 3.1 8B Instruct weights'""",
    f"""CREATE COMPUTE POOL IF NOT EXISTS {CPU_POOL} MIN_NODES = 1 MAX_NODES = 1 INSTANCE_FAMILY = HIGHMEM_X64_S
        AUTO_SUSPEND_SECS = 300 INITIALLY_SUSPENDED = TRUE COMMENT = 'Banking77 WSP: merge edited weights'""",
]


def session():
    from snowflake.snowpark import Session
    return Session.builder.configs({"connection_name": SF_CONNECTION, "authenticator": "snowflake",
                                    "database": SF_DB, "schema": SF_SCHEMA, "warehouse": SF_WH}).create()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--eai", default=EAI, help="external access integration allowing Hugging Face egress")
    ap.add_argument("--payload", default="merge_and_log_job.py", help="job script in hosting/ (merge_q4base_job.py for the 4-bit base)")
    a = ap.parse_args()
    s = session()
    if a.setup:
        for q in SETUP_SQL:
            if a.eai != EAI and "INTEGRATION" in q:
                continue
            print(s.sql(q).collect()[0][0])
    packed = Path(a.packed)
    assert packed.suffix == ".npz" and packed.exists(), packed
    dest = f"@{STAGE}/{a.version}"
    r = s.file.put(str(packed), dest, auto_compress=False, overwrite=True)
    print("PUT", packed.name, r[0].status)
    from snowflake.ml.jobs import submit_file
    job = submit_file(
        str(Path(__file__).with_name(a.payload)),
        CPU_POOL,
        stage_name=JOB_STAGE,
        session=s,
        external_access_integrations=[a.eai],
        env_vars={"EDIT_STAGE_FILE": f"{dest}/{packed.name}", "MODEL_NAME": MODEL_NAME, "VERSION_NAME": a.version,
                  "DB": SF_DB, "SCHEMA": SF_SCHEMA},
    )
    print("job:", job.id, flush=True)
    status = job.wait()
    print("status:", status)
    print(job.get_logs()[-6000:])
    if status == "DONE":
        print(json.dumps(job.result(), indent=1))


if __name__ == "__main__":
    main()
