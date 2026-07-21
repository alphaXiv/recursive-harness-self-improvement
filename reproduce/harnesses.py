from __future__ import annotations

import copy
import json
import re
from typing import Any


INITIAL_HARNESS: dict[str, Any] = {
    "name": "H0_general_team",
    "roles": [
        {
            "id": "planner",
            "kind": "adviser",
            "instruction": "Translate the task into a compact implementation plan and identify edge cases.",
            "contract": ["required_files", "public_api", "edge_cases"],
        },
        {
            "id": "builder",
            "kind": "builder",
            "instruction": "Implement the complete repository from the task and adviser memo.",
            "contract": ["files", "summary"],
        },
        {
            "id": "reviewer",
            "kind": "reviewer",
            "instruction": "Review the draft for correctness, completeness, and reproducibility.",
            "contract": ["blocking_issues", "suggested_fixes"],
        },
    ],
    "hops": [
        {"from": "orchestrator", "to": "planner", "purpose": "plan"},
        {"from": "planner", "to": "builder", "purpose": "handoff plan"},
        {"from": "builder", "to": "reviewer", "purpose": "review draft"},
        {"from": "reviewer", "to": "builder", "purpose": "one repair pass"},
    ],
    "repair_rounds": 1,
    "gates": ["repository JSON parses", "required files are present"],
}


STATIC_HIGH_EFFORT_HARNESS: dict[str, Any] = {
    "name": "static_high_effort",
    "roles": [
        {
            "id": "architect",
            "kind": "adviser",
            "instruction": "Design the public API and implementation architecture before coding.",
            "contract": ["public_api", "module_plan", "invariants"],
        },
        {
            "id": "edge_case_analyst",
            "kind": "adviser",
            "instruction": "Enumerate boundary conditions and adversarial inputs implied by the task.",
            "contract": ["edge_cases", "error_semantics", "test_ideas"],
        },
        {
            "id": "builder",
            "kind": "builder",
            "instruction": "Implement a complete polished repository using all adviser memos.",
            "contract": ["files", "summary"],
        },
        {
            "id": "correctness_reviewer",
            "kind": "reviewer",
            "instruction": "Audit algorithms and edge cases without assuming undocumented behavior.",
            "contract": ["blocking_issues", "counterexamples", "suggested_fixes"],
        },
        {
            "id": "reproducibility_reviewer",
            "kind": "reviewer",
            "instruction": "Audit packaging, documentation, deterministic behavior, and task coverage.",
            "contract": ["blocking_issues", "missing_files", "suggested_fixes"],
        },
    ],
    "hops": [
        {"from": "orchestrator", "to": "architect", "purpose": "architecture"},
        {"from": "orchestrator", "to": "edge_case_analyst", "purpose": "edge cases"},
        {"from": "architect", "to": "builder", "purpose": "structured handoff"},
        {"from": "edge_case_analyst", "to": "builder", "purpose": "structured handoff"},
        {"from": "builder", "to": "correctness_reviewer", "purpose": "independent review"},
        {"from": "builder", "to": "reproducibility_reviewer", "purpose": "independent review"},
        {"from": "correctness_reviewer", "to": "builder", "purpose": "repair"},
        {"from": "reproducibility_reviewer", "to": "builder", "purpose": "repair"},
    ],
    "repair_rounds": 1,
    "gates": ["repository JSON parses", "required files are present", "review blockers addressed"],
}


def harness_text(harness: dict[str, Any]) -> str:
    return json.dumps(harness, sort_keys=True, indent=2)


def validate_harness(candidate: Any, previous: dict[str, Any], iteration: int) -> tuple[dict[str, Any], str]:
    """Constrain optimizer output to the executable prompt-harness schema."""
    if not isinstance(candidate, dict):
        return deterministic_fallback(previous, iteration), "fallback_non_object"
    roles = candidate.get("roles")
    hops = candidate.get("hops")
    if not isinstance(roles, list) or not isinstance(hops, list):
        return deterministic_fallback(previous, iteration), "fallback_missing_structure"

    cleaned_roles = []
    counts = {"adviser": 0, "builder": 0, "reviewer": 0}
    for raw in roles:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("kind", "")).lower()
        if kind not in counts:
            continue
        if kind == "builder" and counts[kind] >= 1:
            continue
        if kind in {"adviser", "reviewer"} and counts[kind] >= 2:
            continue
        role_id = re.sub(r"[^a-z0-9_]+", "_", str(raw.get("id", kind)).lower()).strip("_")
        contract = raw.get("contract", [])
        if not isinstance(contract, list):
            contract = [str(contract)]
        cleaned_roles.append(
            {
                "id": role_id or kind,
                "kind": kind,
                "instruction": str(raw.get("instruction", ""))[:800],
                "contract": [str(x)[:120] for x in contract[:8]],
            }
        )
        counts[kind] += 1
    if counts["builder"] != 1 or not cleaned_roles:
        return deterministic_fallback(previous, iteration), "fallback_no_builder"

    cleaned_hops = []
    for raw in hops[:10]:
        if isinstance(raw, dict):
            cleaned_hops.append(
                {
                    "from": str(raw.get("from", "orchestrator"))[:80],
                    "to": str(raw.get("to", "builder"))[:80],
                    "purpose": str(raw.get("purpose", "handoff"))[:240],
                }
            )
    cleaned = {
        "name": f"H{iteration}_rhi",
        "roles": cleaned_roles,
        "hops": cleaned_hops,
        "repair_rounds": min(1, max(0, int(candidate.get("repair_rounds", 1)))),
        "gates": [str(x)[:200] for x in candidate.get("gates", [])[:8]],
    }
    return cleaned, "model"


def deterministic_fallback(previous: dict[str, Any], iteration: int) -> dict[str, Any]:
    result = copy.deepcopy(previous)
    result["name"] = f"H{iteration}_fallback"
    for role in result.get("roles", []):
        if role.get("kind") == "reviewer":
            role["contract"] = list(dict.fromkeys(role.get("contract", []) + ["evidence", "owner", "retest"]))
            role["instruction"] += " Cite concrete files and route each blocker to an owner with a retest."
        if role.get("kind") == "builder":
            role["contract"] = list(dict.fromkeys(role.get("contract", []) + ["requirements_traceability"]))
    result.setdefault("hops", []).append(
        {"from": "reviewer", "to": "builder", "purpose": "targeted recall with evidence and retest"}
    )
    result["repair_rounds"] = 1
    return result


def structural_metrics(harness: dict[str, Any], task_text: str) -> dict[str, float | int]:
    roles = harness.get("roles", [])
    contract_fields = [field for role in roles for field in role.get("contract", [])]
    words = set(re.findall(r"[a-z][a-z0-9_]{2,}", task_text.lower()))
    harness_words = set(re.findall(r"[a-z][a-z0-9_]{2,}", harness_text(harness).lower()))
    overlap = len(words & harness_words) / max(1, len(words))
    return {
        "roles": len(roles),
        "contract_fields": len(contract_fields),
        "hops": len(harness.get("hops", [])),
        "gates": len(harness.get("gates", [])),
        "task_lexical_overlap": round(overlap, 6),
    }


def semantic_jaccard(a: dict[str, Any], b: dict[str, Any]) -> float:
    wa = set(re.findall(r"[a-z][a-z0-9_]{2,}", harness_text(a).lower()))
    wb = set(re.findall(r"[a-z][a-z0-9_]{2,}", harness_text(b).lower()))
    return round(len(wa & wb) / max(1, len(wa | wb)), 6)
