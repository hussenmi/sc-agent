#!/usr/bin/env python3
"""Compare vision-sidecar models on real scagent figures.

Exercises the SAME code path scagent uses (scagent.agent.vision_sidecar.VisionSidecar),
so a PASS here means the sidecar integration works end-to-end, not just raw HTTP.

Point it at one OpenAI-compatible vision endpoint to smoke-test, or two to A/B
(e.g. Nemotron-Nano-VL NIM vs Qwen3.6-27B). Prints each model's structured
description per figure so you can judge faithfulness on UMAPs / QC violins / dotplots.

Examples:
  # smoke test the Nemotron VL NIM on a couple of figures
  python experiments/vision_ab.py \
    --model nemotron-nano-12b-v2-vl --base-url http://iscn011:8003/v1 --api-key dummy \
    --figures run_*/figures/qc_violin_metrics.png run_*/figures/umap_leiden.png

  # A/B: Nemotron VL vs Qwen3.6-27B
  python experiments/vision_ab.py \
    --model nemotron-nano-12b-v2-vl --base-url http://iscn011:8003/v1 --api-key dummy \
    --model-b Qwen3.6-27B --base-url-b http://iscp001:8000/v1 --api-key-b dummy \
    --figures run_*/figures/qc_violin_metrics.png
"""

from __future__ import annotations

import argparse
import sys
import time

from scagent.agent.vision_sidecar import (
    VisionSidecar,
    VisionSidecarConfig,
    encode_image_path_to_b64,
)


def _sidecar(model: str, base_url: str, api_key: str) -> VisionSidecar:
    return VisionSidecar(
        VisionSidecarConfig(model=model, api_key=api_key or "dummy", base_url=base_url)
    )


def _image_dict(path: str) -> dict:
    b64, mime = encode_image_path_to_b64(path)
    return {"base64": b64, "mime": mime, "path": path, "role": "figure"}


def _describe(sc: VisionSidecar, img: dict) -> tuple[str, int]:
    t0 = time.time()
    r = sc.describe([img])
    dt = int((time.time() - t0) * 1000)
    if r.get("status") != "ok":
        return f"[ERROR: {r.get('error')}]", dt
    return r.get("text", ""), dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", nargs="+", required=True, help="figure PNG paths")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--model-b", default=None, help="optional second model for A/B")
    ap.add_argument("--base-url-b", default=None)
    ap.add_argument("--api-key-b", default="dummy")
    args = ap.parse_args()

    a = _sidecar(args.model, args.base_url, args.api_key)
    b = (
        _sidecar(args.model_b, args.base_url_b, args.api_key_b)
        if args.model_b and args.base_url_b
        else None
    )

    for path in args.figures:
        print("\n" + "=" * 100)
        print(f"FIGURE: {path}")
        print("=" * 100)
        txt_a, dt_a = _describe(a, _image_dict(path))
        print(f"\n----- A: {args.model}  ({dt_a} ms) -----\n{txt_a}")
        if b is not None:
            txt_b, dt_b = _describe(b, _image_dict(path))
            print(f"\n----- B: {args.model_b}  ({dt_b} ms) -----\n{txt_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
