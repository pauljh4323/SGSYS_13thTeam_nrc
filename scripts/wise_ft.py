"""WiSE-FT weight interpolation (Wortsman et al. 2022) between two gru_mask checkpoints of the same architecture.

theta = (1 - alpha) * theta_a + alpha * theta_b over every float tensor of model_state (parameters and normalisation
buffers such as feat_mean / feat_std); non-float tensors are taken from checkpoint a. Keys, shapes, dtypes and
model_cfg must match exactly. The blend is computed in float64 and cast back, so alpha 0 / 1 reproduce a / b exactly.
The output is the usual inference checkpoint (format gru_mask_v1: model_cfg + model_state + str/float metadata),
loadable with torch.load(..., weights_only=True) and nrc.models.gru_mask.load_weights.

Usage (repo root):
  python scripts/wise_ft.py --a weights/gru_v8p2z3e.pt --b weights/gru_v8p2z6a.pt --alpha 0.5 --out weights/x.pt
  python scripts/wise_ft.py --a ... --b ... --sanity 20     # alpha 0 / 1 vs a / b outputs on 20 validation files
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def blend(ck_a: dict, ck_b: dict, alpha: float) -> dict:
    if ck_a.get("format") != ck_b.get("format"):
        raise ValueError(f"format {ck_a.get('format')} != {ck_b.get('format')}")
    if ck_a["model_cfg"] != ck_b["model_cfg"]:
        raise ValueError(f"model_cfg {ck_a['model_cfg']} != {ck_b['model_cfg']}")
    sa, sb = ck_a["model_state"], ck_b["model_state"]
    if list(sa) != list(sb):
        raise ValueError(f"key mismatch: only a {set(sa) - set(sb)}, only b {set(sb) - set(sa)}")
    out = {}
    for k, va in sa.items():
        vb = sb[k]
        if va.shape != vb.shape or va.dtype != vb.dtype:
            raise ValueError(f"{k}: {tuple(va.shape)} {va.dtype} != {tuple(vb.shape)} {vb.dtype}")
        if va.is_floating_point():
            out[k] = ((1.0 - alpha) * va.double() + alpha * vb.double()).to(va.dtype).contiguous()
        else:
            out[k] = va.clone()
    return {"format": ck_a["format"], "model_cfg": dict(ck_a["model_cfg"]), "model_state": out}


def sanity(a: Path, b: Path, n: int) -> bool:
    """alpha 0 / 1 must give bit-identical waveforms to a / b (rms_norm inference) on n validation files."""
    import tempfile
    from nrc.io import load_manifest, read_row, resolve_data_root, stratified_subset
    from nrc.methods import get_method

    root = resolve_data_root(None)
    man = stratified_subset(load_manifest(root, "validation"), n)
    params = dict(device="cpu", level_norm=True, level_mode="rms_norm", target_rms=0.079130)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        for alpha, ref in ((0.0, a), (1.0, b)):
            p = Path(td) / f"wise_{alpha:.1f}.pt"
            torch.save(blend(load(a), load(b), alpha), p)
            sd_ref, sd_bl = load(ref)["model_state"], load(p)["model_state"]
            same_w = all(torch.equal(sd_ref[k], sd_bl[k]) for k in sd_ref)
            m_ref = get_method("gru_mask", weights=str(ref), **params)
            m_bl = get_method("gru_mask", weights=str(p), **params)
            nbad = 0
            for _, row in man.iterrows():
                noisy, sr = read_row(root, row, "noisy"), int(row["sample_rate_hz"])
                y0, y1 = np.asarray(m_ref(noisy, sr)), np.asarray(m_bl(noisy, sr))  # float (pre-PCM) outputs
                nbad += int(not (y0.shape == y1.shape and np.array_equal(y0, y1)))
            print(f"alpha {alpha:.1f} vs {ref.name}: weights equal {same_w}, outputs bit-identical "
                  f"{len(man) - nbad}/{len(man)}")
            ok &= same_w and nbad == 0
    print("SANITY", "PASS" if ok else "FAIL")
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="alpha = 0 end (e.g. the base checkpoint)")
    ap.add_argument("--b", type=Path, required=True, help="alpha = 1 end (the fine-tuned checkpoint)")
    ap.add_argument("--alpha", type=float, nargs="+", help="one or more alphas")
    ap.add_argument("--out", type=str, help="output path; with several alphas use a {a} placeholder, e.g. w_a{a}.pt")
    ap.add_argument("--sanity", type=int, default=0, help="run the alpha 0/1 check on N validation files")
    args = ap.parse_args(argv)
    if args.sanity:
        return 0 if sanity(args.a, args.b, args.sanity) else 1
    if not args.alpha or not args.out:
        ap.error("--alpha and --out are required unless --sanity")
    ck_a, ck_b = load(args.a), load(args.b)
    for alpha in args.alpha:
        out = Path(args.out.format(a=f"{round(alpha * 100):02d}"))
        ck = blend(ck_a, ck_b, alpha)
        ck["wise_ft"] = {"alpha": float(alpha), "a": args.a.as_posix(), "b": args.b.as_posix(),
                         "a_sha256": sha256(args.a), "b_sha256": sha256(args.b)}
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ck, out)
        load(out)  # weights_only round trip
        print(f"alpha {alpha:.2f} -> {out.as_posix()} sha256 {sha256(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
