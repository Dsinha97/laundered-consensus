"""Modal app for the fallback fine-tune (sprint work, 6 Nov 2026 or later). Smoke ladder (download, envcheck, onestep), then sft, dpo and evaluate.

Keys: none needed (both models are open, Apache-2.0, not gated). Volumes: swarm-models (weights), swarm-train (adapters, results).

  PYTHONUTF8=1 .venv/Scripts/modal run train-modal/modal_train.py::download --model qwen9b
  PYTHONUTF8=1 .venv/Scripts/modal run train-modal/modal_train.py::envcheck --model qwen9b
  PYTHONUTF8=1 .venv/Scripts/modal run train-modal/modal_train.py::onestep_qwen
  PYTHONUTF8=1 .venv/Scripts/modal run train-modal/modal_train.py::onestep_gemma

No held-out file is shipped in this image. The stand-in rows are plans/prompts/train_test30.jsonl (Haiku-made, training format).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REMOTE = "/root/repo"

app = modal.App("swarm-train")
models_vol = modal.Volume.from_name("swarm-models", create_if_missing=True)
runs_vol = modal.Volume.from_name("swarm-train", create_if_missing=True)

cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub[hf_transfer]")
gpu_image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch", "transformers", "trl", "peft", "accelerate", "datasets", "huggingface_hub")
             .add_local_dir(str(HERE / "config"), f"{REMOTE}/train-modal/config")
             .add_local_file(str(ROOT / "plans/prompts/prompts/prompt_plain.txt"), f"{REMOTE}/plans/prompts/prompts/prompt_plain.txt")
             .add_local_file(str(ROOT / "plans/prompts/train_test30.jsonl"), f"{REMOTE}/plans/prompts/train_test30.jsonl")
             .add_local_file(str(ROOT / "con-artist/parse.py"), f"{REMOTE}/con-artist/parse.py"))


def load_cfg(name: str) -> dict:
    return json.loads(Path(f"{REMOTE}/train-modal/config/{name}.json").read_text(encoding="utf-8"))


def cfg_local(name: str) -> dict:
    return json.loads((HERE / "config" / f"{name}.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- download (CPU)
@app.function(image=cpu_image.add_local_dir(str(HERE / "config"), f"{REMOTE}/train-modal/config"),
              volumes={"/models": models_vol}, timeout=60 * 60 * 3)
def download_impl(name: str) -> dict:
    import os
    from huggingface_hub import HfApi, snapshot_download

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    cfg = load_cfg(name)
    sha = HfApi().model_info(cfg["model_id"]).sha
    dst = Path("/models") / cfg["volume_dir"]
    snapshot_download(repo_id=cfg["model_id"], revision=sha, local_dir=str(dst),
                      allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "tokenizer*"])
    (dst / "REVISION.txt").write_text(sha, encoding="utf-8")
    files = [p for p in dst.rglob("*") if p.is_file()]
    gb = sum(p.stat().st_size for p in files if p.suffix == ".safetensors") / 1e9
    models_vol.commit()
    return {"model": cfg["model_id"], "revision": sha, "files": len(files), "safetensors_gb": round(gb, 1)}


@app.local_entrypoint()
def download(model: str):
    print(download_impl.remote(model))


# ---------------------------------------------------------------- envcheck (small GPU, tiny random model)
def _envcheck(name: str) -> dict:
    import torch
    import transformers
    import peft
    import trl

    out = {"versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                        "trl": trl.__version__}, "steps": {}}
    cfg = load_cfg(name)
    path = f"/models/{cfg['volume_dir']}"

    def step(label, fn):
        try:
            out["steps"][label] = fn()
        except Exception as e:  # report every failure; do not stop the ladder
            out["steps"][label] = f"FAILED {type(e).__name__}: {str(e)[:300]}"

    tok = {}

    def tokenizer_and_template():
        from transformers import AutoTokenizer
        tok["t"] = AutoTokenizer.from_pretrained(path)
        rows = [json.loads(line) for line in open(f"{REMOTE}/plans/prompts/train_test30.jsonl", encoding="utf-8")]
        system = Path(f"{REMOTE}/plans/prompts/prompts/prompt_plain.txt").read_text(encoding="utf-8-sig")
        kw = {} if cfg["enable_thinking"] is None else {"enable_thinking": cfg["enable_thinking"]}
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": rows[0]["packet_text"]}]
        text = tok["t"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        lens = sorted(len(tok["t"](f"{r['packet_text']}\n{r['chosen']}").input_ids) + 90 for r in rows)
        return {"rendered_head": text[:160], "rendered_tail": text[-120:], "think_block_in_prompt": "<think>" in text,
                "len_p50": lens[len(lens) // 2], "len_p99": lens[int(len(lens) * 0.99) - 1], "len_max": lens[-1]}

    step("tokenizer_and_chat_template", tokenizer_and_template)

    def tiny_model_lora_step():
        from transformers import AutoConfig, AutoModelForCausalLM
        c = AutoConfig.from_pretrained(path)
        tc = getattr(c, "text_config", c)
        tc.num_hidden_layers = 2
        try:
            m = AutoModelForCausalLM.from_config(c, dtype=torch.bfloat16)
        except Exception:
            from transformers import AutoModelForImageTextToText
            m = AutoModelForImageTextToText.from_config(c, dtype=torch.bfloat16)
        m = m.cuda()
        names = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
        pat = re.compile(cfg["lora"]["exclude_regex"])
        targets = sorted({n for n in names if n.split(".")[-1] in cfg["lora"]["target_modules"] and not pat.search(n)})
        lc = peft.LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                             target_modules=targets, task_type="CAUSAL_LM")
        pm = peft.get_peft_model(m, lc)
        trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
        ids = torch.randint(0, 1000, (1, 64), device="cuda")
        loss = pm(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        return {"class": type(m).__name__, "linear_modules": len(names), "lora_targets": len(targets),
                "sample_targets": targets[:3], "trainable_params": trainable, "loss": float(loss.detach())}

    step("tiny_model_lora_forward_backward", tiny_model_lora_step)

    def trl_classes():
        from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer  # noqa: F401
        import inspect
        return {"DPOConfig_has_model_adapter_name": "model_adapter_name" in inspect.signature(DPOConfig).parameters
                or "model_adapter_name" in inspect.signature(DPOTrainer.__init__).parameters,
                "DPOTrainer_params": [p for p in inspect.signature(DPOTrainer.__init__).parameters][:14]}

    step("trl_classes", trl_classes)
    return json.loads(json.dumps(out, default=str))  # plain types only: torch objects cannot be unpickled locally


@app.function(image=gpu_image, gpu="L40S", volumes={"/models": models_vol}, timeout=60 * 30)
def envcheck_impl(name: str) -> dict:
    return _envcheck(name)


@app.local_entrypoint()
def envcheck(model: str):
    print(json.dumps(envcheck_impl.remote(model), indent=2))


# ---------------------------------------------------------------- onestep (real model, memory and speed)
def _onestep(name: str, price_per_hour: float | None) -> dict:
    import torch
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_cfg(name)
    path = f"/models/{cfg['volume_dir']}"
    tok = AutoTokenizer.from_pretrained(path)
    rows = [json.loads(line) for line in open(f"{REMOTE}/plans/prompts/train_test30.jsonl", encoding="utf-8")][:8]
    system = Path(f"{REMOTE}/plans/prompts/prompts/prompt_plain.txt").read_text(encoding="utf-8-sig")
    kw = {} if cfg["enable_thinking"] is None else {"enable_thinking": cfg["enable_thinking"]}
    t0 = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})
    load_s = time.time() - t0
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    pat = re.compile(cfg["lora"]["exclude_regex"])
    targets = sorted({n for n in names if n.split(".")[-1] in cfg["lora"]["target_modules"] and not pat.search(n)})
    lc = peft.LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                         target_modules=targets, task_type="CAUSAL_LM")
    pm = peft.get_peft_model(model, lc)
    pm.gradient_checkpointing_enable()
    pm.enable_input_require_grads()
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)

    def encode(r, completion):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["packet_text"]}]
        prompt_ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False, **kw)
        comp_ids = tok(completion + (tok.eos_token or ""), add_special_tokens=False).input_ids
        ids = list(prompt_ids) + comp_ids
        labels = [-100] * len(prompt_ids) + comp_ids
        return torch.tensor([ids], device="cuda"), torch.tensor([labels], device="cuda")

    opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-5)
    torch.cuda.reset_peak_memory_stats()
    times, seqlens = [], []
    pm.train()
    for r in rows[:4]:  # SFT-style steps, batch size 1
        ids, labels = encode(r, r["chosen"])
        seqlens.append(int(ids.shape[1]))
        torch.cuda.synchronize()
        s = time.time()
        loss = pm(input_ids=ids, labels=labels).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        times.append(time.time() - s)
    sft_peak = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    # DPO-shaped step: policy forward with grad on chosen and rejected, plus a no-grad reference forward (adapter off)
    dpo_times = []
    for r in rows[4:7]:
        ci, cl = encode(r, r["chosen"])
        ri, rl = encode(r, r["rejected"])
        torch.cuda.synchronize()
        s = time.time()
        with torch.no_grad(), pm.disable_adapter():
            pm(input_ids=ci, labels=cl)
            pm(input_ids=ri, labels=rl)
        loss = pm(input_ids=ci, labels=cl).loss + pm(input_ids=ri, labels=rl).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
        torch.cuda.synchronize()
        dpo_times.append(time.time() - s)
    dpo_peak = torch.cuda.max_memory_allocated() / 1e9
    gpu = torch.cuda.get_device_name(0)
    mean = lambda xs: sum(xs[1:]) / max(1, len(xs) - 1)  # skip the first (warm-up) step
    res = {"model": cfg["model_id"], "gpu": gpu, "load_seconds": round(load_s, 1), "lora_targets": len(targets),
           "trainable_params": trainable, "seq_lens": seqlens,
           "sft_step_seconds": [round(x, 2) for x in times], "sft_peak_gb": round(sft_peak, 1),
           "dpo_step_seconds": [round(x, 2) for x in dpo_times], "dpo_peak_gb": round(dpo_peak, 1),
           "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)}
    if price_per_hour:
        res["forecast_usd_per_1000_sft_examples"] = round(mean(times) * 1000 / 3600 * price_per_hour, 2)
        res["forecast_usd_per_1000_dpo_pairs"] = round(mean(dpo_times) * 1000 / 3600 * price_per_hour, 2)
    Path("/runs").mkdir(exist_ok=True)
    Path(f"/runs/onestep_{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    runs_vol.commit()
    return json.loads(json.dumps(res, default=str))


@app.function(image=gpu_image, gpu="H200", volumes={"/models": models_vol, "/runs": runs_vol}, timeout=60 * 40)
def onestep_gemma_impl(price_per_hour: float | None = None) -> dict:
    return _onestep("gemma31b", price_per_hour)


@app.function(image=gpu_image, gpu="H100", volumes={"/models": models_vol, "/runs": runs_vol}, timeout=60 * 40)
def onestep_qwen_impl(price_per_hour: float | None = None) -> dict:
    return _onestep("qwen9b", price_per_hour)


@app.local_entrypoint()
def onestep_gemma(price_per_hour: float = 0.0):
    print(json.dumps(onestep_gemma_impl.remote(price_per_hour or None), indent=2))


@app.local_entrypoint()
def onestep_qwen(price_per_hour: float = 0.0):
    print(json.dumps(onestep_qwen_impl.remote(price_per_hour or None), indent=2))


# ================================================================ training and evaluation (frozen runs only)
# Plain PyTorch loops (no TRL), so nothing depends on TRL API changes. Run A = SFT; Run B = DPO that starts from the Run A
# adapter, with reference log-probs computed once from that same starting adapter (so policy and reference begin equal).
# Held-out v2 is never shipped to a training function. The evaluate function receives eval packets as an argument only.
import hashlib as _hl  # noqa: E402


def _sha(p) -> str:
    return _hl.sha256(Path(p).read_bytes()).hexdigest()


def _load_model_and_tok(cfg, adapter_dir=None, trainable=True):
    import torch
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = f"/models/{cfg['volume_dir']}"
    if cfg.get("revision"):  # the weights on the volume must be the pinned revision
        on_disk = Path(path, "REVISION.txt").read_text(encoding="utf-8").strip()
        if on_disk != cfg["revision"]:
            raise RuntimeError(f"weights on the volume are revision {on_disk}, config pins {cfg['revision']}")
    tok = AutoTokenizer.from_pretrained(path)
    try:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})
    except Exception:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})
    if adapter_dir == "base":
        return model, tok
    if adapter_dir:
        pm = peft.PeftModel.from_pretrained(model, adapter_dir, is_trainable=trainable)
    else:
        names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        pat = re.compile(cfg["lora"]["exclude_regex"])
        targets = sorted({n for n in names if n.split(".")[-1] in cfg["lora"]["target_modules"] and not pat.search(n)})
        lc = peft.LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"], lora_dropout=cfg["lora"]["dropout"],
                             target_modules=targets, task_type="CAUSAL_LM")
        pm = peft.get_peft_model(model, lc)
    if trainable:
        pm.gradient_checkpointing_enable()
        pm.enable_input_require_grads()
    return pm, tok


def _encode(tok, cfg, prompt_msgs, completion):
    kw = {} if cfg["enable_thinking"] is None else {"enable_thinking": cfg["enable_thinking"]}
    p = tok.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True, return_dict=False, **kw)
    c = tok(completion + (tok.eos_token or ""), add_special_tokens=False).input_ids
    return list(p), c


def _completion_logp(pm, p_ids, c_ids):
    """Sum and count of log-probs of the completion tokens given the prompt."""
    import torch
    ids = torch.tensor([p_ids + c_ids], device="cuda")
    logits = pm(input_ids=ids).logits[0, len(p_ids) - 1:-1].float()
    lp = torch.log_softmax(logits, dim=-1).gather(-1, torch.tensor(c_ids, device="cuda")[:, None])[:, 0]
    return lp.sum(), lp.numel()


def _train(kind: str, name: str, run: str, train_rows: list, dev_rows: list, freeze_sha: str, start_adapter: str | None,
           max_seconds: int) -> dict:
    import math
    import random
    import torch
    from transformers import get_cosine_schedule_with_warmup

    cfg = load_cfg(name)
    hp = cfg[kind]
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    pm, tok = _load_model_and_tok(cfg, adapter_dir=start_adapter, trainable=True)
    pm.train()

    def prep(rows):
        out, dropped = [], 0
        for r in rows:
            pc, cc = _encode(tok, cfg, r["prompt"], (r["completion"] if kind == "sft" else r["chosen"])[0]["content"])
            if kind == "dpo":
                pr, cr = _encode(tok, cfg, r["prompt"], r["rejected"][0]["content"])
                if len(pc) + max(len(cc), len(cr)) > cfg["max_len_cap"]:
                    dropped += 1
                    continue
                out.append((r["id"], pc, cc, cr))
            else:
                if len(pc) + len(cc) > cfg["max_len_cap"]:
                    dropped += 1
                    continue
                out.append((r["id"], pc, cc))
        return out, dropped

    tr, dropped = prep(train_rows)
    dv, _ = prep(dev_rows)
    if dropped / max(1, len(train_rows)) > cfg["max_dropped_fraction"]:
        raise RuntimeError(f"{dropped} of {len(train_rows)} rows exceed max_len_cap; stopping")
    random.shuffle(tr)
    accum = hp["effective_batch"]
    steps = math.ceil(len(tr) * hp["epochs"] / accum)
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(hp.get("warmup_ratio", 0.03) * steps), steps)

    ref = {}
    if kind == "dpo":  # reference = the starting adapter, computed once with dropout off and no gradient
        pm.eval()
        with torch.no_grad():
            for rid, pc, cc, cr in tr + dv:
                ref[rid] = (float(_completion_logp(pm, pc, cc)[0]), float(_completion_logp(pm, pc, cr)[0]))
        pm.train()

    def loss_of(item):
        if kind == "sft":
            _, pc, cc = item
            s, n = _completion_logp(pm, pc, cc)
            return -(s / n), None
        rid, pc, cc, cr = item
        sc, _ = _completion_logp(pm, pc, cc)
        sr, _ = _completion_logp(pm, pc, cr)
        rc, rr = ref[rid]
        margin = hp["beta"] * ((sc - rc) - (sr - rr))
        return -torch.nn.functional.logsigmoid(margin), float(margin.detach() > 0)

    t0 = time.time()
    log, order = [], list(range(len(tr))) * math.ceil(hp["epochs"])
    stopped = None
    for step in range(steps):
        batch = [tr[i] for i in order[step * accum:(step + 1) * accum]]
        tot, accs = 0.0, []
        for item in batch:
            loss, acc = loss_of(item)
            (loss / len(batch)).backward()
            tot += float(loss.detach()) / len(batch)
            if acc is not None:
                accs.append(acc)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        log.append({"step": step + 1, "loss": round(tot, 4), "pref_acc": round(sum(accs) / len(accs), 3) if accs else None,
                    "seconds": round(time.time() - t0, 1)})
        if time.time() - t0 > max_seconds:
            stopped = f"time cap {max_seconds}s reached at step {step + 1} of {steps}"
            break
    pm.eval()
    dev_loss = []
    with torch.no_grad():
        for item in dv:
            l, _ = loss_of(item)
            dev_loss.append(float(l))
    adapter_dir = f"/runs/{run}/adapter"
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(adapter_dir)
    res = {"run": run, "kind": kind, "model": cfg["model_id"], "freeze_sha256": freeze_sha, "start_adapter": start_adapter,
           "train_rows": len(tr), "dropped_over_cap": dropped, "dev_rows": len(dv), "steps_planned": steps,
           "steps_done": len(log), "stopped_early": stopped, "train_seconds": round(time.time() - t0, 1),
           "dev_loss_mean": round(sum(dev_loss) / max(1, len(dev_loss)), 4), "log": log,
           "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "hp": hp}
    Path(f"/runs/{run}/train_result.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    runs_vol.commit()
    return json.loads(json.dumps(res, default=str))


def _evaluate(name: str, run: str, adapter: str, rows: list, max_new_tokens: int, freeze_sha: str) -> dict:
    """Greedy generation under PLAIN, one packet at a time. Output holds raw text; scoring is done locally by score_eval.py."""
    import torch
    import importlib.util

    spec = importlib.util.spec_from_file_location("ca_parse", f"{REMOTE}/con-artist/parse.py")
    parse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parse)
    cfg = load_cfg(name)
    pm, tok = _load_model_and_tok(cfg, adapter_dir=("base" if adapter == "base" else f"/runs/{adapter}/adapter"), trainable=False)
    pm.eval()
    system = Path(f"{REMOTE}/plans/prompts/prompts/prompt_plain.txt").read_text(encoding="utf-8-sig")
    kw = {} if cfg["enable_thinking"] is None else {"enable_thinking": cfg["enable_thinking"]}
    out = []
    t0 = time.time()
    for r in rows:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["packet_text"]}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=False, **kw)
        x = torch.tensor([list(ids)], device="cuda")
        with torch.no_grad():
            g = pm.generate(input_ids=x, max_new_tokens=max_new_tokens, do_sample=False)
        text = tok.decode(g[0, x.shape[1]:], skip_special_tokens=True)
        v = parse.parse_strict(text)
        lv = parse.parse_lenient(text) if hasattr(parse, "parse_lenient") else None
        rec = {k: r[k] for k in r if k != "packet_text"}
        rec.update({"raw_output": text, "parse_ok": v is not None, "verdict": v["verdict"] if v else None,
                    "lenient_verdict": lv.get("verdict") if isinstance(lv, dict) else None,
                    "model": adapter, "not_for_training": True})
        out.append(rec)
    res = {"run": run, "adapter": adapter, "freeze_sha256": freeze_sha, "n": len(out), "seconds": round(time.time() - t0, 1)}
    Path(f"/runs/{run}").mkdir(parents=True, exist_ok=True)
    Path(f"/runs/{run}/eval_{adapter.replace('/', '_')}.jsonl").write_text(
        "\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    runs_vol.commit()
    return {"summary": res, "records": out}


def _gpu_fn(gpu):
    return app.function(image=gpu_image, gpu=gpu, volumes={"/models": models_vol, "/runs": runs_vol}, timeout=60 * 60 * 4)


@_gpu_fn("H200")
def train_gemma_impl(kind, run, train_rows, dev_rows, freeze_sha, start_adapter, max_seconds):
    return _train(kind, "gemma31b", run, train_rows, dev_rows, freeze_sha, start_adapter, max_seconds)


@_gpu_fn("H100")
def train_qwen_impl(kind, run, train_rows, dev_rows, freeze_sha, start_adapter, max_seconds):
    return _train(kind, "qwen9b", run, train_rows, dev_rows, freeze_sha, start_adapter, max_seconds)


@_gpu_fn("H200")
def eval_gemma_impl(run, adapter, rows, max_new_tokens, freeze_sha):
    return _evaluate("gemma31b", run, adapter, rows, max_new_tokens, freeze_sha)


@_gpu_fn("H100")
def eval_qwen_impl(run, adapter, rows, max_new_tokens, freeze_sha):
    return _evaluate("qwen9b", run, adapter, rows, max_new_tokens, freeze_sha)


def _local_freeze_check(run: str) -> str:
    """Verify the freeze file on the laptop; refuse to launch on any difference. Returns the freeze file's sha256."""
    import sys
    sys.path.insert(0, str(HERE))
    import freeze_train
    fp = HERE / "freezes" / f"{run}.json"
    if not fp.exists():
        raise SystemExit(f"REFUSED: no freeze file {fp}")
    bad = freeze_train.verify(fp)
    if bad:
        raise SystemExit("REFUSED: freeze check failed:\n  " + "\n  ".join(bad))
    return _sha(fp)


def _jsonl(p):
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def _run_train(kind, model, run, data_dir, max_seconds, start_run=None):
    fsha = _local_freeze_check(run)
    d = Path(data_dir)
    fn = train_gemma_impl if model == "gemma31b" else train_qwen_impl
    res = fn.remote(kind, run, _jsonl(d / f"train_{kind}.jsonl"), _jsonl(d / f"dev_{kind}.jsonl"), fsha,
                    f"/runs/{start_run}/adapter" if start_run else None, max_seconds)
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"{run}_train_result.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "log"}, indent=2), "\nfull log:", out)


@app.local_entrypoint()
def sft(model: str, run: str, data_dir: str = "train-modal/data", max_seconds: int = 7200):
    """Run A. Needs freezes/<run>.json written by freeze_train.py after prep and leak_check."""
    _run_train("sft", model, run, data_dir, max_seconds)


@app.local_entrypoint()
def dpo(model: str, run: str, start_run: str, data_dir: str = "train-modal/data", max_seconds: int = 7200):
    """Run B: starts from the adapter of Run A (start_run on the swarm-train volume)."""
    _run_train("dpo", model, run, data_dir, max_seconds, start_run)


@app.local_entrypoint()
def evaluate(model: str, run: str, adapter: str, eval_file: str, expected_sha: str, max_new_tokens: int = 400,
             limit: int = 0):
    """adapter: 'base' or a run name whose adapter is on the volume. eval_file is read here, its sha256 must equal
    expected_sha, and its rows go to the function as an argument only (never into the image)."""
    fsha = _local_freeze_check(run)
    if _sha(eval_file) != expected_sha.lower():
        raise SystemExit("REFUSED: the eval file's sha256 differs from the expected one")
    rows = _jsonl(eval_file)
    if limit:
        rows = rows[:limit]
    fn = eval_gemma_impl if model == "gemma31b" else eval_qwen_impl
    res = fn.remote(run, adapter, rows, max_new_tokens, fsha)
    (HERE / "results").mkdir(exist_ok=True)
    out = HERE / "results" / f"{run}_eval_{adapter}.jsonl"
    out.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in res["records"]) + "\n", encoding="utf-8")
    print(json.dumps(res["summary"], indent=2), "\nrecords:", out)
