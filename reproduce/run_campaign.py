from __future__ import annotations

import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import pathlib
import random
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

from .harnesses import (
    INITIAL_HARNESS,
    STATIC_HIGH_EFFORT_HARNESS,
    harness_text,
    semantic_jaccard,
    structural_metrics,
    validate_harness,
)
from .tasks import RUNNER, TASKS, Task


ROOT = pathlib.Path(__file__).resolve().parents[1]


def extract_json(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def safe_repo(candidate: Any) -> dict[str, Any] | None:
    if not isinstance(candidate, dict) or not isinstance(candidate.get("files"), dict):
        return None
    files: dict[str, str] = {}
    for raw_path, content in candidate["files"].items():
        path = pathlib.PurePosixPath(str(raw_path))
        if path.is_absolute() or ".." in path.parts or len(path.parts) > 5:
            continue
        if not isinstance(content, str) or len(content) > 60_000:
            continue
        files[str(path)] = content
    if "solution.py" not in files:
        return None
    return {"files": files, "summary": str(candidate.get("summary", ""))[:2000]}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    peak_context_tokens: int = 0
    reused_prefix_tokens_proxy: int = 0
    previous_prompt_ids: list[int] = field(default_factory=list)

    def add(self, prompt_ids: list[int], output_tokens: int) -> None:
        lcp = 0
        for a, b in zip(self.previous_prompt_ids, prompt_ids):
            if a != b:
                break
            lcp += 1
        self.input_tokens += len(prompt_ids)
        self.output_tokens += output_tokens
        self.calls += 1
        self.peak_context_tokens = max(self.peak_context_tokens, len(prompt_ids) + output_tokens)
        self.reused_prefix_tokens_proxy += lcp
        self.previous_prompt_ids = prompt_ids

    def public(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "peak_context_tokens": self.peak_context_tokens,
            "cache_counters_exposed": False,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "reused_prefix_tokens_proxy": self.reused_prefix_tokens_proxy,
        }


class LocalModel:
    def __init__(self, model_path: str, gpu: int, config: dict[str, Any], worker_seed: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": f"cuda:{gpu}"},
            attn_implementation="sdpa",
        )
        self.model.eval()
        self.temperature = float(config["temperature"])
        self.top_p = float(config["top_p"])
        self.worker_seed = worker_seed
        self.call_index = 0

    def call(self, system: str, user: str, max_new_tokens: int, usage: Usage) -> str:
        self.call_index += 1
        seed = self.worker_seed * 10000 + self.call_index
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(rendered, return_tensors="pt", truncation=True, max_length=24576)
        prompt_ids = encoded.input_ids[0].tolist()
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=max(self.temperature, 1e-5),
                top_p=self.top_p,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        continuation = generated[0, encoded["input_ids"].shape[1] :]
        usage.add(prompt_ids, int(continuation.shape[0]))
        return self.tokenizer.decode(continuation, skip_special_tokens=True)


def compact_repo(repo: dict[str, Any], limit: int = 24_000) -> str:
    parts = []
    for path, content in sorted(repo.get("files", {}).items()):
        parts.append(f"FILE {path}\n{content[:8000]}")
    return "\n\n".join(parts)[:limit]


def generate_repo(model: LocalModel, task: Task, harness: dict[str, Any], label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = Usage()
    memos = []
    roles = harness.get("roles", [])
    for role in [r for r in roles if r.get("kind") == "adviser"]:
        response = model.call(
            "You are a specialist adviser inside a coding-agent harness. Return concise JSON only.",
            f"TASK:\n{task.prompt}\n\nROLE:\n{json.dumps(role, sort_keys=True)}\n\n"
            "Return an evidence-grounded memo satisfying the role's contract. Do not write the repository.",
            1100,
            usage,
        )
        memos.append({"role": role.get("id"), "memo": response[:6000]})

    builder = next((r for r in roles if r.get("kind") == "builder"), {"id": "builder", "instruction": "Implement."})
    response = model.call(
        "You are the implementation agent. Return valid repository JSON only, with no markdown fences or commentary.",
        f"TASK:\n{task.prompt}\n\nFULL HARNESS:\n{harness_text(harness)}\n\nADVISER MEMOS:\n{json.dumps(memos)}\n\n"
        f"BUILDER ROLE:\n{json.dumps(builder)}\n\nReturn {{\"files\": {{...}}, \"summary\": \"...\"}}.",
        5200,
        usage,
    )
    try:
        repo = safe_repo(extract_json(response))
    except Exception:
        repo = None
    if repo is None:
        retry = model.call(
            "Repair malformed output. Return valid repository JSON only.",
            f"TASK:\n{task.prompt}\n\nMALFORMED OUTPUT:\n{response[:18000]}\n\n"
            "Return exactly an object with files and summary; solution.py is mandatory.",
            5200,
            usage,
        )
        try:
            repo = safe_repo(extract_json(retry))
        except Exception:
            repo = None
    if repo is None:
        repo = {"files": {"solution.py": "# generation failed\n", "README.md": "Generation failed.\n", "pyproject.toml": "[project]\nname='failed'\nversion='0.0.0'\n"}, "summary": "invalid generation"}

    reviewers = [r for r in roles if r.get("kind") == "reviewer"]
    critiques = []
    for role in reviewers:
        critique = model.call(
            "You are an independent repository reviewer. Return concise JSON only.",
            f"TASK:\n{task.prompt}\n\nREVIEW ROLE:\n{json.dumps(role)}\n\nREPOSITORY:\n{compact_repo(repo)}\n\n"
            "Find concrete blocking issues and fixes. Do not rewrite the repository.",
            1200,
            usage,
        )
        critiques.append({"role": role.get("id"), "critique": critique[:7000]})

    for _ in range(int(harness.get("repair_rounds", 0))):
        if not critiques:
            break
        patched = model.call(
            "You are the implementation agent performing a targeted repair. Return complete valid repository JSON only.",
            f"TASK:\n{task.prompt}\n\nCURRENT REPOSITORY:\n{compact_repo(repo)}\n\nREVIEWS:\n{json.dumps(critiques)}\n\n"
            "Return the complete repository after fixing valid blockers. Preserve correct behavior.",
            5200,
            usage,
        )
        try:
            maybe = safe_repo(extract_json(patched))
        except Exception:
            maybe = None
        if maybe is not None:
            repo = maybe

    return repo, {"label": label, **usage.public(), "file_count": len(repo["files"]), "chars": sum(len(v) for v in repo["files"].values())}


def generate_plain(model: LocalModel, task: Task) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = Usage()
    out = model.call(
        "You are a coding agent. Return valid repository JSON only, with no markdown fences.",
        task.prompt + '\nReturn {"files": {...}, "summary": "..."}.',
        5200,
        usage,
    )
    try:
        repo = safe_repo(extract_json(out))
    except Exception:
        repo = None
    if repo is None:
        repo = {"files": {"solution.py": "# invalid\n"}, "summary": "invalid"}
    return repo, {"label": "plain", **usage.public(), "file_count": len(repo["files"]), "chars": sum(len(v) for v in repo["files"].values())}


def evaluate_repo(task: Task, repo: dict[str, Any]) -> dict[str, Any]:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="rhi-artifact-") as tmp:
        root = pathlib.Path(tmp)
        for rel, content in repo["files"].items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        hidden = root / "hidden_tests"
        hidden.mkdir()
        (hidden / "test_case.py").write_text(task.tests, encoding="utf-8")
        (hidden / "runner.py").write_text(RUNNER, encoding="utf-8")
        try:
            proc = subprocess.run(
                ["python", str(hidden / "runner.py")],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=25,
                env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
            )
            match = re.search(r"ORX_TEST_RESULT=(\{.*\})", proc.stdout)
            result = json.loads(match.group(1)) if match else {"total": 0, "passed": 0, "failures": 0, "errors": 1}
            details = [json.loads(x) for x in re.findall(r"ORX_TEST_DETAIL=(\{.*\})", proc.stdout)]
        except subprocess.TimeoutExpired:
            result = {"total": 0, "passed": 0, "failures": 0, "errors": 1}
            details = [{"test": "timeout", "tail": "exceeded 25 seconds"}]
    required = ["solution.py", "README.md", "pyproject.toml"]
    coverage = sum(path in repo["files"] for path in required) / len(required)
    executable = result["passed"] / max(1, result["total"])
    return {
        "tests_total": result["total"],
        "tests_passed": result["passed"],
        "executable_score": round(executable, 6),
        "deliverable_coverage": round(coverage, 6),
        "fixed_score": round(0.85 * executable + 0.15 * coverage, 6),
        "failure_summaries": details[:4],
        "elapsed_seconds": round(time.time() - started, 3),
    }


JUDGE_SYSTEM = """You are a strict open-model evaluator comparing two anonymous repositories for the same task.
Judge task compliance, likely functional correctness, edge-case handling, reproducibility, documentation,
and engineering quality. Artifact labels are random and carry no condition information. Choose left, right,
or tie. Return JSON only with keys winner, rationale, left_strengths, right_strengths, and actionable_feedback.
Do not infer hidden tests and do not prefer an artifact merely because it is longer."""


def one_judgment(model: LocalModel, task: Task, left: dict[str, Any], right: dict[str, Any], usage: Usage) -> dict[str, Any]:
    left_id = hashlib.sha256(compact_repo(left).encode()).hexdigest()[:10]
    right_id = hashlib.sha256(compact_repo(right).encode()).hexdigest()[:10]
    prompt = (
        f"TASK:\n{task.prompt}\n\nLEFT ARTIFACT {left_id}:\n{compact_repo(left)}\n\n"
        f"RIGHT ARTIFACT {right_id}:\n{compact_repo(right)}\n\nReturn the required JSON."
    )
    raw = model.call(JUDGE_SYSTEM, prompt, 650, usage)
    try:
        parsed = extract_json(raw)
    except Exception:
        parsed = {"winner": "tie", "rationale": "judge parse failure", "actionable_feedback": ["Return valid JSON"]}
    winner = str(parsed.get("winner", "tie")).lower()
    if winner not in {"left", "right", "tie"}:
        winner = "tie"
    parsed["winner"] = winner
    return parsed


def pairwise(model: LocalModel, task: Task, a: dict[str, Any], b: dict[str, Any], name_a: str, name_b: str) -> dict[str, Any]:
    usage = Usage()
    forward = one_judgment(model, task, a, b, usage)
    reverse = one_judgment(model, task, b, a, usage)
    f = name_a if forward["winner"] == "left" else name_b if forward["winner"] == "right" else "tie"
    r = name_b if reverse["winner"] == "left" else name_a if reverse["winner"] == "right" else "tie"
    consensus = f if f == r else "tie"
    return {
        "a": name_a,
        "b": name_b,
        "forward_mapped": f,
        "reverse_mapped": r,
        "consensus": consensus,
        "order_consistent": f == r,
        "forward_rationale": str(forward.get("rationale", ""))[:1200],
        "reverse_rationale": str(reverse.get("rationale", ""))[:1200],
        "actionable_feedback": [
            str(x)[:500]
            for parsed in (forward, reverse)
            for x in parsed.get("actionable_feedback", [])[:4]
        ][:8],
        "judge_usage": usage.public(),
    }


def optimize_harness(
    model: LocalModel,
    task: Task,
    previous: dict[str, Any],
    previous_repo: dict[str, Any],
    previous_eval: dict[str, Any],
    history: list[dict[str, Any]],
    iteration: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    usage = Usage()
    system = """You are a principal prompt engineer for autonomous coding agents. Improve only the
multi-agent harness. Prioritize task-specific structured output contracts and explicit orchestrator-agent
hops with critique, targeted recall, and repair. Ground every change in the artifact evidence and cumulative
pairwise history. You are not given the evaluator rubric. Return one JSON harness only. It must contain name,
roles, hops, repair_rounds, and gates. Each role has id, kind (adviser, builder, or reviewer), instruction, and
contract (a list of fields). Include exactly one builder, at most two advisers, at most two reviewers, at most
ten hops, and repair_rounds must be 0 or 1."""
    user = (
        f"TASK:\n{task.prompt}\n\nCURRENT HARNESS:\n{harness_text(previous)}\n\n"
        f"CURRENT ARTIFACT EXCERPT:\n{compact_repo(previous_repo, 14000)}\n\n"
        f"EXECUTION EVIDENCE:\n{json.dumps(previous_eval)}\n\n"
        f"CUMULATIVE PAIRWISE HISTORY:\n{json.dumps(history)}\n\n"
        f"Produce H{iteration} as a full replacement harness. Preserve working traits and fix evidenced defects."
    )
    raw = model.call(system, user, 2600, usage)
    try:
        candidate = extract_json(raw)
    except Exception:
        candidate = None
    harness, source = validate_harness(candidate, previous, iteration)
    return harness, {"iteration": iteration, "source": source, **usage.public()}


def run_task(model: LocalModel, task: Task, seed: int) -> dict[str, Any]:
    task_started = time.time()
    plain, plain_usage = generate_plain(model, task)
    e_plain = evaluate_repo(task, plain)

    h0 = copy.deepcopy(INITIAL_HARNESS)
    r0, u0 = generate_repo(model, task, h0, "H0")
    e0 = evaluate_repo(task, r0)
    bootstrap = pairwise(model, task, r0, plain, "H0", "plain")
    history = [bootstrap]

    h1, oh1 = optimize_harness(model, task, h0, r0, e0, history, 1)
    r1, u1 = generate_repo(model, task, h1, "H1")
    e1 = evaluate_repo(task, r1)
    local1 = pairwise(model, task, r1, r0, "H1", "H0")
    history.append(local1)

    h2, oh2 = optimize_harness(model, task, h1, r1, e1, history, 2)
    r2, u2 = generate_repo(model, task, h2, "H2")
    e2 = evaluate_repo(task, r2)

    static, us = generate_repo(model, task, STATIC_HIGH_EFFORT_HARNESS, "static_high_effort")
    es = evaluate_repo(task, static)

    p21 = pairwise(model, task, r2, r1, "H2", "H1")
    p20 = pairwise(model, task, r2, r0, "H2", "H0")
    p2s = pairwise(model, task, r2, static, "H2", "static_high_effort")

    harnesses = {"H0": h0, "H1": h1, "H2": h2, "static_high_effort": STATIC_HIGH_EFFORT_HARNESS}
    result = {
        "task_id": task.id,
        "domain": task.domain,
        "seed": seed,
        "scores": {"plain": e_plain, "H0": e0, "H1": e1, "H2": e2, "static_high_effort": es},
        "preferences": {"H0_vs_plain": bootstrap, "H1_vs_H0": local1, "H2_vs_H1": p21, "H2_vs_H0": p20, "H2_vs_static": p2s},
        "usage": {"plain": plain_usage, "H0": u0, "H1": u1, "H2": u2, "static_high_effort": us},
        "optimizer_usage": {"H1": oh1, "H2": oh2},
        "structure": {name: structural_metrics(h, task.prompt) for name, h in harnesses.items()},
        "semantic_similarity": {"H0_H1": semantic_jaccard(h0, h1), "H1_H2": semantic_jaccard(h1, h2), "H0_H2": semantic_jaccard(h0, h2)},
        "harnesses": harnesses,
        "elapsed_seconds": round(time.time() - task_started, 3),
    }
    return result


def worker_main(gpu: int, task_indices: list[int], model_path: str, config: dict[str, Any], queue: Any) -> None:
    seed = int(config["seed"])
    model = LocalModel(model_path, gpu, config, seed * 100 + gpu)
    for idx in task_indices:
        task = TASKS[idx]
        try:
            queue.put({"type": "progress", "task": task.id, "state": "started", "gpu": gpu})
            result = run_task(model, task, seed)
            queue.put({"type": "result", "result": result, "gpu": gpu})
        except Exception as exc:
            queue.put({"type": "error", "task": task.id, "error": repr(exc), "gpu": gpu})
    queue.put({"type": "worker_done", "gpu": gpu})


def aggregate(results: list[dict[str, Any]], started: float, config: dict[str, Any]) -> dict[str, Any]:
    conditions = ["plain", "H0", "H1", "H2", "static_high_effort"]
    summary: dict[str, Any] = {}
    for condition in conditions:
        rows = [r for result in results for r in [result["scores"][condition]]]
        usages = [result["usage"][condition] for result in results]
        summary[condition] = {
            "n_tasks": len(rows),
            "mean_executable_score": round(sum(r["executable_score"] for r in rows) / max(1, len(rows)), 6),
            "mean_fixed_score": round(sum(r["fixed_score"] for r in rows) / max(1, len(rows)), 6),
            "total_input_tokens": sum(u["input_tokens"] for u in usages),
            "total_output_tokens": sum(u["output_tokens"] for u in usages),
            "mean_peak_context_tokens": round(sum(u["peak_context_tokens"] for u in usages) / max(1, len(usages)), 3),
            "total_calls": sum(u["calls"] for u in usages),
            "cache_counters_exposed": False,
            "total_reused_prefix_tokens_proxy": sum(u["reused_prefix_tokens_proxy"] for u in usages),
        }
    prefs = {}
    for key in ["H1_vs_H0", "H2_vs_H1", "H2_vs_H0", "H2_vs_static"]:
        vals = [r["preferences"][key]["consensus"] for r in results]
        left = results[0]["preferences"][key]["a"] if results else "left"
        right = results[0]["preferences"][key]["b"] if results else "right"
        prefs[key] = {"n": len(vals), left: vals.count(left), right: vals.count(right), "tie": vals.count("tie")}
    return {
        "schema": "rhi-reproduction-result-v1",
        "model_id": config["model_id"],
        "seed": config["seed"],
        "backend": "kubernetes",
        "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
        "allocated_gpu_count": 4,
        "completed_tasks": len(results),
        "wall_seconds": round(time.time() - started, 3),
        "condition_summary": summary,
        "preference_summary": prefs,
        "tasks": sorted(results, key=lambda x: x["task_id"]),
    }


def main() -> None:
    started = time.time()
    config = json.loads((ROOT / "config.json").read_text())
    print("PROTOCOL_CONFIG=" + json.dumps(config, sort_keys=True), flush=True)
    from huggingface_hub import snapshot_download

    model_path = snapshot_download(config["model_id"])
    print("MODEL_SNAPSHOT=" + str(model_path), flush=True)
    task_indices = list(range(len(TASKS))) if config.get("tasks") == "all" else list(config["tasks"])
    workers = min(int(config.get("max_workers", 4)), 4, len(task_indices))
    assignments = [task_indices[i::workers] for i in range(workers)]
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    processes = [ctx.Process(target=worker_main, args=(gpu, assignments[gpu], model_path, config, queue)) for gpu in range(workers)]
    for process in processes:
        process.start()
    results = []
    errors = []
    done = 0
    while done < workers:
        message = queue.get()
        if message["type"] == "progress":
            print("TASK_PROGRESS=" + json.dumps(message, sort_keys=True), flush=True)
        elif message["type"] == "result":
            results.append(message["result"])
            compact = {k: v for k, v in message["result"].items() if k != "harnesses"}
            print("TASK_RESULT=" + json.dumps(compact, sort_keys=True), flush=True)
        elif message["type"] == "error":
            errors.append(message)
            print("TASK_ERROR=" + json.dumps(message, sort_keys=True), flush=True)
        elif message["type"] == "worker_done":
            done += 1
    for process in processes:
        process.join(timeout=30)
    final = aggregate(results, started, config)
    final["errors"] = errors
    print("FINAL_RESULT=" + json.dumps(final, sort_keys=True), flush=True)
    if errors or len(results) != len(task_indices):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
