"""Create the GPU inference service for a registered edited-model version (starts GPU compute: run only
after explicit approval). Run in the background: creation takes 5-15+ minutes.

    python create_service.py --version V_<RUN> [--setup]
--setup creates BANKING77_WSP_GPU_POOL (GPU_NV_S, 1 node, auto-suspend 10 min) if missing.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "harness"))
from common import SF_DB, SF_SCHEMA  # noqa: E402
from submit_merge_job import MODEL_NAME, session  # noqa: E402

GPU_POOL = "BANKING77_WSP_GPU_POOL"
SERVICE = "BANKING77_WSP_SVC"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--service", default=SERVICE)
    ap.add_argument("--setup", action="store_true")
    a = ap.parse_args()
    s = session()
    if a.setup:
        print(s.sql(f"""CREATE COMPUTE POOL IF NOT EXISTS {GPU_POOL} MIN_NODES = 1 MAX_NODES = 1
            INSTANCE_FAMILY = GPU_NV_S AUTO_SUSPEND_SECS = 600 INITIALLY_SUSPENDED = TRUE
            COMMENT = 'Banking77 WSP: serve the edited Llama 3.1 8B'""").collect()[0][0])
    from snowflake.ml.model.inference_engine import InferenceEngine
    from snowflake.ml.registry import Registry
    mv = Registry(session=s, database_name=SF_DB, schema_name=SF_SCHEMA).get_model(MODEL_NAME).version(a.version)
    print("Creating service...", flush=True)
    mv.create_service(
        service_name=f"{SF_DB}.{SF_SCHEMA}.{a.service}",
        service_compute_pool=GPU_POOL,
        ingress_enabled=True,
        max_instances=1,
        gpu_requests="1",
        # A10G has 22 GiB: 16 GB of 16-bit weights + KV cache only fits with a short context.
        # The longest Banking77 prompt is ~145 tokens, so 2,048 is ample. --dtype=auto follows the model config
        # (float16 for the 4-bit-base rebuild, bfloat16 for V_OPUS55_R1).
        inference_engine_options={"engine": InferenceEngine.VLLM,
                                  "engine_args_override": ["--max-model-len=2048", "--gpu-memory-utilization=0.92",
                                                           "--dtype=auto"]},
    )
    print("Service created:", a.service, flush=True)
    print(mv.list_services())


if __name__ == "__main__":
    main()
