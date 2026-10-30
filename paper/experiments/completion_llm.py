"""One bounded Modal invocation; cache each complete parent group locally."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator

import modal

TARGETS = ("AIR BAGS", "EXTERIOR LIGHTING")

app = modal.App("certineg-completion")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.6.0", "transformers==4.51.3", "accelerate==1.6.0"
)


@app.function(
    image=image,
    gpu="T4",
    cpu=(1, 2),
    memory=(8192, 8192),
    max_containers=1,
    timeout=1800,
    startup_timeout=120,
    scaledown_window=2,
)
def score(rows: list[dict], revision: str) -> Iterator[dict]:
    """Stream only complete parents; likelihood scores never see reference labels."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.monotonic()
    torch.set_num_threads(2)
    model_id = "Qwen/Qwen3-0.6B"
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.float16
        )
        .cuda()
        .eval()
    )
    yes = tok.encode("Yes", add_special_tokens=False)
    no = tok.encode("No", add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        raise ValueError("Protocol requires single-token Yes and No")
    tok.padding_side = "left"
    for parent in sorted({r["parent"] for r in rows}):
        block = [r for r in rows if r["parent"] == parent]
        output = []
        for target in TARGETS:
            for start in range(0, len(block), 16):
                if time.monotonic() - started > 1650:
                    yield {
                        "incomplete_parent": parent,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    return
                batch = block[start : start + 16]
                prompts = [
                    tok.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": f"Does this vehicle complaint describe an issue with {target}? Answer Yes or No. Complaint: {r['summary']}",
                            }
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    for r in batch
                ]
                lengths = [len(tok.encode(p)) for p in prompts]
                inputs = tok(
                    prompts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to("cuda")
                with torch.inference_mode():
                    logits = model(**inputs).logits[:, -1, :]
                    probs = (
                        torch.softmax(logits[:, [no[0], yes[0]]].float(), dim=-1)[:, 1]
                        .cpu()
                        .tolist()
                    )
                output.extend(
                    {
                        "complaint_id": r["complaint_id"],
                        "target": target,
                        "score": p,
                        "truncated": n > 512,
                    }
                    for r, p, n in zip(batch, probs, lengths, strict=True)
                )
        yield {
            "parent": parent,
            "scores": output,
            "elapsed_seconds": time.monotonic() - started,
        }


@app.local_entrypoint()
def main() -> None:
    """Persist input provenance before remote execution and stream checkpoints."""
    from completion_data import ROOT, load_data, selected_rows

    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    if (ROOT / "data/results/completion/llm_scores.jsonl").exists():
        raise FileExistsError(
            "Preserve the existing score cache before intentionally launching a new paid run"
        )
    df, tr, te = load_data()
    cal, test = selected_rows(df, tr, 16), selected_rows(df, te, 64)
    import pandas as pd

    chosen = pd.concat([cal.assign(part="calibration"), test.assign(part="test")])
    out = ROOT / "data/results/completion"
    out.mkdir(exist_ok=True)
    chosen[["complaint_id", "parent", "part"]].to_csv(
        out / "llm_membership.csv", index=False
    )
    rows = chosen[["complaint_id", "parent", "summary"]].to_dict("records")
    manifest = {
        "model": "Qwen/Qwen3-0.6B",
        "revision": revision,
        "targets": TARGETS,
        "max_tokens": 512,
        "calibration_parents": 16,
        "test_parents": 64,
        "rows": len(rows),
        "input_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "gpu": "T4",
        "timeout": 1800,
        "retries": 0,
        "rate_usd_second": 0.000164 + 2 * 0.0000131 + 8 * 0.00000222,
    }
    (out / "llm_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (out / "llm_scores.jsonl").open("w") as f:
        for block in score.remote_gen(rows, revision):
            f.write(json.dumps(block) + "\n")
            f.flush()
            print({k: v for k, v in block.items() if k != "scores"}, flush=True)
