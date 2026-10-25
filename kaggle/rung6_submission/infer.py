"""Rung 6 Kaggle submission: Gemma-2-9B-it + LoRA (8-bit, T4 x2), both orders, internet off.

One script, two roles. The main process verifies the attached private dataset against its
MANIFEST, installs the repo wheel offline, writes a prior-filled submission.csv at once (a valid
file exists even if inference later fails), splits the rows over the GPUs and starts one worker
per GPU. Each worker loads its own 8-bit copy of the base model plus the adapter and score head,
runs its share longest-first in token-budget batches (both orders, swap-averaged later), and
writes part{g}.parquet (ids and raw probabilities only) every few batches. A worker projects the
time left from its measured throughput and drops to single-order (ab only) when both orders would
not finish inside the wall-clock budget. The main process merges, optionally applies
calibrator.json, writes submission.csv and run_log.json. No dataset text is ever written.

`--timing-rows N` runs the same pipeline on N cv rows of the competition train.csv and writes
timing.json (a projection to 25,000 rows x 2 orders); it never writes submission.csv.
`--no-quant --allow-no-bos --skip-install` exist only for CPU dry runs on a tiny model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_ROOT = "/kaggle/input"
DEFAULT_OUT = "/kaggle/working"
COMPETITION_HINT = "llm-classification-finetuning"
DEFAULT_LIMIT_S = 8 * 3600 + 15 * 60
IGNORED_EXTRA = {"MANIFEST.json", "dataset-metadata.json"}
# import name -> (pip distribution prefix, needed even with --no-quant)
REQUIRED_MODULES = {
    "sentencepiece": ("sentencepiece", True),
    "peft": ("peft", True),
    "accelerate": ("accelerate", True),
    "bitsandbytes": ("bitsandbytes", False),
}


def log(msg: str) -> None:
    print(f"[infer {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- offline bootstrap (stdlib only)


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_dataset_dir(roots: list[str], explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    found = []
    for r in roots:
        for m in sorted(Path(r).rglob("MANIFEST.json")):
            try:
                if "adapter_tag" in json.loads(m.read_text()):
                    found.append(m.parent)
            except (OSError, ValueError):
                continue
    if len(found) != 1:
        raise SystemExit(f"expected one rung-6 dataset (MANIFEST.json) under {roots}, got {found}")
    return found[0]


def verify_manifest(ds: Path, allow_dev: bool = False) -> dict:
    """Every file in MANIFEST.json must exist with its sha256, and nothing else may be there."""
    manifest = json.loads((ds / "MANIFEST.json").read_text())
    if manifest.get("git_sha") in (None, "", "dev") and not allow_dev:
        raise SystemExit("MANIFEST.json has no pinned git sha (a dev build); refusing")
    want = manifest["files"]
    for rel, meta in want.items():
        p = ds / rel
        if not p.is_file():
            raise SystemExit(f"MANIFEST lists {rel} but it is missing from {ds}")
        got = sha256_path(p)
        if got != meta["sha256"]:
            raise SystemExit(f"{rel}: sha256 {got} != manifest {meta['sha256']}")
    present = {str(p.relative_to(ds)) for p in ds.rglob("*") if p.is_file()}
    extra = present - set(want) - IGNORED_EXTRA
    if extra:
        raise SystemExit(f"files in the dataset that MANIFEST.json does not list: {sorted(extra)}")
    log(f"manifest verified: {len(want)} files, git {str(manifest.get('git_sha'))[:12]}")
    return manifest


def pip_install(args: list[str]) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-index", *args])


def bootstrap(ds: Path, manifest: dict, need_bnb: bool) -> dict:
    """Install the repo wheel and any missing third-party package from the dataset, offline."""
    info = {"installed": [], "missing_after": []}
    wheels = sorted(f for f in manifest["files"] if f.endswith(".whl") and "/" not in f)
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one repo wheel at the dataset root, got {wheels}")
    pip_install(["--no-deps", "--force-reinstall", str(ds / wheels[0])])
    info["installed"].append(wheels[0])
    importlib.invalidate_caches()
    for mod, (dist, always) in REQUIRED_MODULES.items():
        if not (always or need_bnb):
            continue
        if importlib.util.find_spec(mod) is not None:
            continue
        cands = sorted((ds / "wheels").glob(f"{dist}-*.whl")) if (ds / "wheels").is_dir() else []
        if not cands:
            info["missing_after"].append(mod)
            continue
        pip_install(["--no-deps", str(cands[-1])])
        info["installed"].append(f"wheels/{cands[-1].name}")
        importlib.invalidate_caches()
        if importlib.util.find_spec(mod) is None:
            info["missing_after"].append(mod)
    if info["missing_after"]:
        raise SystemExit(
            f"missing packages with no wheel in {ds / 'wheels'}: {info['missing_after']}; rebuild "
            "the dataset with build_dataset.py --include-wheels"
        )
    return info


# ---------------------------------------------------------------- shared small helpers


def versions() -> dict:
    from importlib import metadata

    out = {"python": sys.version.split()[0]}
    for d in ("torch", "transformers", "peft", "bitsandbytes", "accelerate", "numpy", "pandas"):
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out


def write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def adapter_settings(adapter: Path) -> dict:
    meta = adapter / "adapter_meta.json"
    return json.loads(meta.read_text()) if meta.is_file() else {}


# ---------------------------------------------------------------- worker (one per GPU)


def worker_main(cfg: dict) -> int:
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from llm_preference_audit import lora_train as lt
    from llm_preference_audit import submission as S
    from llm_preference_audit.lora_inputs import build_lora_ids, field_window

    g, out = int(cfg["gpu"]), Path(cfg["out_dir"])
    t_start = time.time()
    wlog: dict = {"gpu": g, "t": cfg["t"], "attn": cfg["attn"], "quantized": not cfg["no_quant"]}
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    adt = torch.float16 if use_cuda else torch.bfloat16
    wlog["gpu_name"] = torch.cuda.get_device_name(0) if use_cuda else "cpu"

    ids_wanted = set(json.loads(Path(cfg["shard_file"]).read_text()))
    df = S.load_test(cfg["input_csv"])
    df = df[df["id"].isin(ids_wanted)].reset_index(drop=True)
    results: dict[int, np.ndarray] = {}
    part = out / f"part{g}.parquet"
    if part.is_file():  # a retry after a crash resumes from the last flushed part
        old = pd.read_parquet(part)
        for r in old.to_numpy(dtype=float):
            results[int(r[0])] = r[1:]
        df = df[~df["id"].isin(results)].reset_index(drop=True)
        wlog["resumed_rows"] = len(results)
    wlog["rows_assigned"] = len(ids_wanted)

    tok = AutoTokenizer.from_pretrained(cfg["model_dir"], local_files_only=True)
    tok.model_max_length = 10**9
    if tok.pad_token_id is None:
        raise SystemExit("tokenizer has no pad token")
    if tok.bos_token_id is None and not cfg["allow_no_bos"]:
        raise SystemExit("the Gemma tokenizer must have a BOS token")
    template = lt.tokenizer_template(tok)
    wlog["template_overhead"] = template.overhead

    t0 = time.time()
    enc = S.encode_unlabelled(
        df,
        lambda s: len(tok(s, add_special_tokens=False).input_ids),
        lambda texts: tok(texts, add_special_tokens=False)["input_ids"],
        field_window(cfg["t"], template),
        cfg["middle_share"],
    )
    wlog["encode_s"] = time.time() - t0
    wlog["rows_hard_cut"] = int(enc.hard_cut.sum())

    t0 = time.time()
    kw = {
        "num_labels": 3,
        "attn_implementation": cfg["attn"],
        "pad_token_id": tok.pad_token_id,
        "local_files_only": True,
    }
    if cfg["no_quant"]:
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_dir"], dtype=adt, **kw
        ).to(device)
    else:
        from transformers import BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["score"])
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_dir"], quantization_config=q, dtype=torch.float16, device_map={"": 0}, **kw
        )
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    import peft

    model = peft.PeftModel.from_pretrained(
        model, cfg["adapter_dir"], is_trainable=False, local_files_only=True
    )
    model.eval()
    wlog["model_load_s"] = time.time() - t0
    log(
        f"worker {g}: encoded {len(df)} rows ({wlog['encode_s']:.0f}s), model ready "
        f"({wlog['model_load_s']:.0f}s)"
    )

    pad = tok.pad_token_id
    n_oom = [0]
    slow = float(cfg.get("sleep_s", 0.0))  # dry runs only: make the time projection deterministic

    def probs(seqs):
        try:
            x, mask = lt.collate(seqs, pad, device)
            with torch.no_grad(), torch.autocast(device.type, dtype=adt):
                logits = model(input_ids=x, attention_mask=mask).logits
            if slow:
                time.sleep(slow)
            return torch.softmax(logits.float(), dim=-1).cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if len(seqs) == 1:
                raise
            n_oom[0] += 1
            torch.cuda.empty_cache()
            h = len(seqs) // 2
            return np.vstack([probs(seqs[:h]), probs(seqs[h:])])

    lens = np.array(
        [
            min(cfg["t"], template.overhead + len(p) + len(a) + len(b))
            for p, a, b in zip(enc.p, enc.a, enc.b, strict=True)
        ],
        dtype=np.int64,
    )
    order = np.argsort(-lens, kind="stable")
    sorted_lens = lens[order].tolist()
    batches = S.token_budget_batches(sorted_lens, cfg["batch_tokens"], cfg["max_batch"])
    costs = [S.batch_cost(sorted_lens, b) for b in batches]
    remaining = float(sum(costs))
    deadline = float(cfg["deadline_epoch"])
    done_cost, compute_s, nonfinite, n_rows_by_pass = 0.0, 0.0, 0, {1: 0, 2: 0}
    switched_at, stopped_at = None, None
    nan3 = np.full(3, np.nan)
    t_infer = time.time()

    def flush() -> None:
        ids = sorted(results)
        frame = pd.DataFrame([[i, *results[i]] for i in ids], columns=S.PART_COLUMNS).astype(
            {"id": "int64"}
        )
        S.assert_no_text(frame)
        tmp = part.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, part)

    for bi, b in enumerate(batches):
        time_left = deadline - time.time()
        if time_left <= 0:
            stopped_at = bi
            break
        rate = done_cost / compute_s if compute_s > 0 else None
        passes = S.choose_passes(remaining, rate, time_left, bi)
        if passes == 1 and switched_at is None and bi >= S.WARMUP_BATCHES:
            switched_at = bi
            log(
                f"worker {g}: projected overrun at batch {bi}/{len(batches)}; single order from here"
            )
        rows = [int(order[j]) for j in b]
        t_b = time.time()
        ab_seqs = [build_lora_ids(template, enc.p[i], enc.a[i], enc.b[i], cfg["t"]) for i in rows]
        ab = probs(ab_seqs)
        if passes == 2:
            ba_seqs = [
                build_lora_ids(template, enc.p[i], enc.b[i], enc.a[i], cfg["t"]) for i in rows
            ]
            ba = probs(ba_seqs)
        else:
            ba = np.tile(nan3, (len(rows), 1))
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_s += time.time() - t_b
        done_cost += costs[bi] * passes
        remaining -= costs[bi]
        for k, i in enumerate(rows):
            row_ab, row_ba = ab[k], ba[k]
            if not np.isfinite(row_ab).all():
                row_ab, nonfinite = nan3, nonfinite + 1
            if not np.isfinite(row_ba).all() and passes == 2:
                row_ba, nonfinite = nan3, nonfinite + 1
            results[int(enc.ids[i])] = np.concatenate([row_ab, row_ba])
            n_rows_by_pass[passes] += 1
        if (bi + 1) % cfg["flush_batches"] == 0:
            flush()
            rps = done_cost / max(compute_s, 1e-9)
            log(
                f"worker {g}: batch {bi + 1}/{len(batches)} rows {len(results)} "
                f"{rps:.0f} padded-tok/s/pass left {max(deadline - time.time(), 0) / 60:.0f}m"
            )
    flush()
    infer_s = time.time() - t_infer
    done_rows = sum(
        len(batches[j]) for j in range(len(batches)) if stopped_at is None or j < stopped_at
    )
    wlog |= {
        "infer_s": infer_s,
        "compute_s": compute_s,
        "rows_done": done_rows,
        "rows_both_orders": n_rows_by_pass[2],
        "rows_single_order": n_rows_by_pass[1],
        "rows_not_reached": len(df) - done_rows,
        "switched_to_single_at_batch": switched_at,
        "n_batches": len(batches),
        "nonfinite_rows": nonfinite,
        "oom_splits": n_oom[0],
        "padded_tokens_per_s_per_pass": done_cost / compute_s if compute_s else None,
        "padded_tokens_both_passes": done_cost,
        "peak_gib": torch.cuda.max_memory_allocated() / 1024**3 if use_cuda else None,
        "total_s": time.time() - t_start,
    }
    write_json(out / f"worker{g}.json", wlog)
    log(f"worker {g}: done {done_rows}/{len(df)} rows, {n_rows_by_pass[1]} single-order")
    return 0


# ---------------------------------------------------------------- main process


def parse(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", default="", help=argparse.SUPPRESS)
    ap.add_argument("--roots", default=DEFAULT_ROOT, help="comma-separated input roots")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--dataset-dir", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--model-hint", default="9b-it")
    ap.add_argument("--t", type=int, default=None, help="default: adapter_meta.json, else 2048")
    ap.add_argument("--middle-share", type=float, default=None)
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="eager")
    ap.add_argument("--batch-tokens", type=int, default=4096, help="padded tokens per batch")
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--flush-batches", type=int, default=50)
    ap.add_argument("--gpus", type=int, default=0, help="0 = all visible CUDA devices")
    ap.add_argument("--limit-s", type=float, default=DEFAULT_LIMIT_S)
    ap.add_argument("--reserve-s", type=float, default=300.0, help="kept for merging and writing")
    ap.add_argument("--timing-rows", type=int, default=0)
    ap.add_argument("--target-rows", type=int, default=25_000)
    ap.add_argument("--max-missing-frac", type=float, default=0.02)
    ap.add_argument("--no-quant", action="store_true", help="dry run only: no 8-bit")
    ap.add_argument("--allow-no-bos", action="store_true", help="dry run only (Qwen tokenizer)")
    ap.add_argument("--skip-install", action="store_true", help="dry run only: no pip install")
    ap.add_argument("--allow-dev-manifest", action="store_true", help="dry run only")
    ap.add_argument(
        "--dry-run-sleep-s", type=float, default=0.0, help="dry run only: sleep per pass"
    )
    return ap.parse_args(argv)


def spawn_workers(a, cfgs: list[dict], n_cuda: int) -> list[int]:
    """Start one worker per GPU, wait, and rerun a failed one once (it resumes from its part)."""
    codes = []
    procs = {}
    for cfg in cfgs:
        env = dict(os.environ)
        if n_cuda >= 2:
            env["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
        procs[cfg["gpu"]] = (cfg, env)
    running = {}
    for g, (cfg, env) in procs.items():
        running[g] = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", cfg["_file"]],
            env=env,
        )
    for g, p in running.items():
        code = p.wait()
        if code != 0 and time.time() < cfgs[0]["deadline_epoch"]:
            log(f"worker {g} exited {code}; rerunning once (resumes from its part file)")
            cfg, env = procs[g]
            code = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--config",
                    cfg["_file"],
                ],
                env=env,
            ).wait()
        codes.append(code)
    return codes


def main(argv=None) -> int:
    a = parse(argv)
    if a.worker:
        cfg = json.loads(Path(a.config).read_text())
        return worker_main(cfg)

    t_main = time.time()
    phases: dict[str, float] = {}
    roots = [r for r in a.roots.split(",") if r]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timing = a.timing_rows > 0

    t0 = time.time()
    ds = find_dataset_dir(roots, a.dataset_dir)
    manifest = verify_manifest(ds, a.allow_dev_manifest)
    boot = {"installed": [], "skipped": True}
    if not a.skip_install:
        boot = bootstrap(ds, manifest, need_bnb=not a.no_quant)
    phases["bootstrap_s"] = time.time() - t0

    import numpy as np
    import pandas as pd

    from llm_preference_audit import submission as S
    from llm_preference_audit.kernel_common import TRAIN_SHA256, load_verified
    from llm_preference_audit.lora_train import SAMPLE_SEED

    adapter = ds / "adapter"
    meta = adapter_settings(adapter)
    t_len = a.t or int(meta.get("t", 2048))
    ms = a.middle_share if a.middle_share is not None else float(meta.get("middle_share", 0.2))
    if a.t and "t" in meta and int(meta["t"]) != a.t:
        log(f"WARNING: --t {a.t} differs from the adapter's training T={meta['t']}")
    model_dir = Path(a.model_dir) if a.model_dir else S.locate_model_dir(roots, hint=a.model_hint)
    log(f"model {model_dir}; adapter {adapter}; T={t_len}; {'timing' if timing else 'submission'}")

    work = Path(tempfile.mkdtemp(prefix="rung6_"))  # texts for the workers; never under out_dir
    try:
        t0 = time.time()
        prior = None
        train_csv = None
        try:
            train_csv = S.locate_input("train.csv", roots, expect_sha=TRAIN_SHA256)
            prior = S.class_prior(train_csv)
        except FileNotFoundError as e:
            log(f"WARNING: no verified train.csv ({e}); the fallback prior is uniform")
        prior_vec = prior if prior is not None else np.full(3, 1 / 3)
        labels = None
        if timing:
            if train_csv is None:
                raise SystemExit("timing mode needs the competition train.csv")
            df_all, manifest_df = load_verified(train_csv.parent, TRAIN_SHA256)
            cv_ids = np.sort(manifest_df.loc[manifest_df["split"] == "cv", "id"].to_numpy())
            pick = np.random.default_rng(SAMPLE_SEED).choice(
                cv_ids, size=min(a.timing_rows, len(cv_ids)), replace=False
            )
            raw = pd.read_csv(train_csv)
            raw = raw[raw["id"].isin(pick)]
            lab = df_all.set_index("id")["label"]
            labels = {int(i): ["a", "b", "tie"].index(str(lab.loc[i])) for i in raw["id"]}
            input_csv = work / "input.csv"
            raw[list(S.TEST_COLUMNS)].to_csv(input_csv, index=False)
            del df_all, raw
        else:
            test_csv = S.locate_input("test.csv", roots, prefer=COMPETITION_HINT)
            input_csv = test_csv
        data = S.load_test(input_csv)
        ids = data["id"].to_numpy()
        n_rows = len(ids)
        if not timing:  # (a) a valid file exists before any GPU work
            S.prior_submission(ids.tolist(), prior_vec).to_csv(out / "submission.csv", index=False)
            log(f"wrote prior-filled submission.csv ({n_rows} rows) before inference")
        # character length is the cheap proxy for token load (capped at the model window)
        proxy = [
            min(t_len, (sum(map(len, p)) + sum(map(len, r1)) + sum(map(len, r2))) / 3.5)
            for p, r1, r2 in zip(
                data["prompt"], data["response_a"], data["response_b"], strict=True
            )
        ]
        del data
        phases["prepare_s"] = time.time() - t0

        n_cuda = 0
        try:
            import torch

            n_cuda = torch.cuda.device_count()
        except ImportError:
            pass
        n_gpu = a.gpus or max(n_cuda, 1)
        shares = S.split_for_gpus(proxy, n_gpu)
        deadline = t_main + a.limit_s - a.reserve_s
        cfgs = []
        for g, idx in enumerate(shares):
            shard = work / f"shard{g}.json"
            shard.write_text(json.dumps([int(ids[i]) for i in idx]))
            cfg = {
                "gpu": g, "out_dir": str(out), "input_csv": str(input_csv),
                "shard_file": str(shard), "model_dir": str(model_dir), "adapter_dir": str(adapter),
                "t": t_len, "middle_share": ms, "attn": a.attn, "batch_tokens": a.batch_tokens,
                "max_batch": a.max_batch, "flush_batches": a.flush_batches, "no_quant": a.no_quant,
                "allow_no_bos": a.allow_no_bos, "deadline_epoch": deadline,
                "sleep_s": a.dry_run_sleep_s,
            }  # fmt: skip
            cfg_file = work / f"cfg{g}.json"
            cfg_file.write_text(json.dumps(cfg))
            cfgs.append(cfg | {"_file": str(cfg_file)})
        for g in range(n_gpu):  # stale parts from an earlier run would be resumed as results
            (out / f"part{g}.parquet").unlink(missing_ok=True)
            (out / f"worker{g}.json").unlink(missing_ok=True)
        log(
            f"{n_rows} rows over {n_gpu} worker(s); deadline in {(deadline - time.time()) / 3600:.2f}h"
        )
        t0 = time.time()
        codes = spawn_workers(a, cfgs, n_cuda)
        phases["workers_wall_s"] = time.time() - t0

        t0 = time.time()
        parts = [pd.read_parquet(out / f"part{g}.parquet") for g in range(n_gpu)
                 if (out / f"part{g}.parquet").is_file()]  # fmt: skip
        merged = S.merge_parts(parts)
        ids_out, P, filled = S.fill_missing(merged, ids.tolist(), prior_vec)
        coef = None
        cal_file = ds / "calibrator.json"
        if cal_file.is_file() and not timing:
            coef = json.loads(cal_file.read_text())
            P = np.where(filled[:, None], P, S.apply_calibrator(P, coef))
        wl = [json.loads((out / f"worker{g}.json").read_text())
              for g in range(n_gpu) if (out / f"worker{g}.json").is_file()]  # fmt: skip
        n_by_orders = {
            int(k): int(v)
            for k, v in zip(*np.unique(merged.n_orders, return_counts=True), strict=True)
        }
        n_by_orders[0] = n_by_orders.get(0, 0) + int(n_rows - len(merged.ids))
        run = {
            "mode": "timing" if timing else "submission",
            "n_rows": n_rows, "rows_by_orders_run": n_by_orders,
            "rows_single_order": int(sum(w.get("rows_single_order", 0) for w in wl)),
            "rows_filled_with_prior": int(filled.sum()),
            "nonfinite_rows": int(sum(w.get("nonfinite_rows", 0) for w in wl)),
            "worker_exit_codes": codes, "calibrated": coef is not None,
            "t": t_len, "middle_share": ms, "attn": a.attn, "quantized": not a.no_quant,
            "n_gpu": n_gpu, "limit_s": a.limit_s,
            "adapter_tag": manifest.get("adapter_tag"), "git_sha": manifest.get("git_sha"),
            "wheel": next((f for f in manifest["files"] if f.endswith(".whl") and "/" not in f), None),
            "bootstrap": boot, "versions": versions(), "workers": wl,
            "prior_is_uniform_fallback": prior is None,
        }  # fmt: skip
        if timing:
            P_ok = ~filled
            if P_ok.any():
                from llm_preference_audit.scoring import log_loss

                y = np.array([labels[int(i)] for i in ids_out[P_ok]])
                run["timing_in_sample_log_loss"] = log_loss(y, P[P_ok])  # cv rows: sanity only
            infer_s = max((w["infer_s"] for w in wl), default=0.0)
            if infer_s > 0 and len(wl) == n_gpu:
                run["timing"] = S.project_timing(
                    n_rows=n_rows, infer_s=infer_s,
                    encode_s=max(w["encode_s"] for w in wl),
                    load_s=max(w["model_load_s"] for w in wl),
                    overhead_s=phases["bootstrap_s"] + phases["prepare_s"],
                    n_target=a.target_rows, limit_s=a.limit_s,
                )  # fmt: skip
                write_json(out / "timing.json", run["timing"] | {
                    "peak_gib_per_gpu": [w.get("peak_gib") for w in wl],
                    "padded_tokens_per_s_per_pass": [w.get("padded_tokens_per_s_per_pass") for w in wl],
                    "model_load_s": [w["model_load_s"] for w in wl],
                    "encode_s": [w["encode_s"] for w in wl],
                    "infer_s": [w["infer_s"] for w in wl],
                    "nonfinite_rows": run["nonfinite_rows"],
                    "oom_splits": [w.get("oom_splits") for w in wl],
                    "timing_in_sample_log_loss": run.get("timing_in_sample_log_loss"),
                    "human": S.human_seconds(run["timing"]["projected_total_with_safety_s"]),
                })  # fmt: skip
        else:
            sub = S.submission_frame(ids_out.tolist(), P, ids.tolist())
            S.assert_no_text(sub)
            sub.to_csv(out / "submission.csv", index=False)
        phases["merge_s"] = time.time() - t0
        phases["total_s"] = time.time() - t_main
        run["phases_s"] = phases
        write_json(out / "run_log.json", run)
        missing_frac = run["rows_filled_with_prior"] / max(n_rows, 1)
        log(
            f"done: {n_rows} rows, {run['rows_filled_with_prior']} prior-filled, "
            f"{run['rows_single_order']} single-order, {phases['total_s'] / 60:.1f} min"
        )
        if any(codes) or missing_frac > a.max_missing_frac:
            log(f"FAILING LOUD: worker codes {codes}, {missing_frac:.1%} of rows prior-filled")
            return 3
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
