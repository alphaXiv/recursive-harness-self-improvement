from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    domain: str
    prompt: str
    tests: str


COMMON = """
Build a small, production-quality Python repository using only the standard library.
Return a JSON object with exactly two top-level keys: `files` (a mapping from relative paths to
complete file contents) and `summary`. Required paths are solution.py, README.md, and
pyproject.toml. Do not include binary files, shell commands, markdown fences, or absolute paths.
Public APIs must be deterministic and documented. The repository will be checked by unseen
executable tests derived only from this task statement.
""".strip()


TASKS = [
    Task(
        "data_ledger",
        "data_infrastructure",
        COMMON + """

Implement `summarize_ledger(rows)` where rows is an iterable of dictionaries containing
`account`, `amount`, and `currency`. Amount accepts int, float, or a decimal string. Currency is
case-insensitive and must be normalized to uppercase. Return a dictionary with `balances` and
`rejected`: balances maps currency then account to a two-decimal string, sorted by currency and
account; rejected is a list of zero-based input indices whose row is missing a field, has an empty
account/currency, a non-finite amount, or an invalid numeric amount. Never mutate input rows.
Use decimal arithmetic and half-even rounding.
""",
        r'''
import copy, math, unittest
import solution

class Tests(unittest.TestCase):
    def test_basic(self):
        rows=[{"account":"a","amount":"1.235","currency":"usd"},{"account":"a","amount":2,"currency":"USD"},{"account":"b","amount":"-0.105","currency":"eur"}]
        self.assertEqual(solution.summarize_ledger(rows), {"balances":{"EUR":{"b":"-0.10"},"USD":{"a":"3.24"}},"rejected":[]})
    def test_rejected(self):
        rows=[{"account":"","amount":1,"currency":"USD"},{"account":"a","amount":"nan","currency":"USD"},{"account":"a","currency":"USD"},{"account":"a","amount":"x","currency":"USD"}]
        self.assertEqual(solution.summarize_ledger(rows)["rejected"],[0,1,2,3])
    def test_no_mutation(self):
        rows=[{"account":"x","amount":"2.00","currency":"gbp"}]; old=copy.deepcopy(rows)
        solution.summarize_ledger(rows); self.assertEqual(rows,old)
    def test_empty(self): self.assertEqual(solution.summarize_ledger([]),{"balances":{},"rejected":[]})
    def test_sorting(self):
        r=solution.summarize_ledger([{"account":"z","amount":1,"currency":"usd"},{"account":"a","amount":1,"currency":"eur"}])
        self.assertEqual(list(r["balances"]),["EUR","USD"])
''',
    ),
    Task(
        "data_events",
        "data_infrastructure",
        COMMON + """

Implement `compact_events(events)` for an iterable of dictionaries with `id`, `timestamp`, and
`payload`. Timestamp is an ISO-8601 string; accept a trailing `Z` as UTC. For each non-empty id,
retain the event with the latest instant. On equal instants, retain the later input occurrence.
Return new dictionaries ordered by ascending instant and then id. Normalize returned timestamps
to UTC `YYYY-MM-DDTHH:MM:SSZ`, deep-copy payloads, preserve no extra keys, and raise ValueError
for a missing/empty id or an invalid/naive timestamp. Do not mutate the input.
""",
        r'''
import copy, unittest
import solution

class Tests(unittest.TestCase):
    def test_latest_and_order(self):
        e=[{"id":"a","timestamp":"2024-01-01T00:00:00Z","payload":{"v":1}},{"id":"b","timestamp":"2023-12-31T19:00:01-05:00","payload":2},{"id":"a","timestamp":"2024-01-02T00:00:00+00:00","payload":{"v":3}}]
        r=solution.compact_events(e); self.assertEqual([x["id"] for x in r],["b","a"]); self.assertEqual(r[1]["payload"],{"v":3})
    def test_tie_later(self):
        r=solution.compact_events([{"id":"x","timestamp":"2024-01-01T00:00:00Z","payload":1},{"id":"x","timestamp":"2023-12-31T19:00:00-05:00","payload":2}])
        self.assertEqual(r[0]["payload"],2)
    def test_normalized(self): self.assertEqual(solution.compact_events([{"id":"x","timestamp":"2024-01-01T01:02:03+00:00","payload":0}])[0]["timestamp"],"2024-01-01T01:02:03Z")
    def test_invalid(self):
        for event in [{"id":"","timestamp":"2024-01-01T00:00:00Z","payload":0},{"id":"x","timestamp":"2024-01-01T00:00:00","payload":0}]:
            with self.assertRaises(ValueError): solution.compact_events([event])
    def test_deepcopy(self):
        e=[{"id":"x","timestamp":"2024-01-01T00:00:00Z","payload":{"a":[]},"extra":1}]; old=copy.deepcopy(e); r=solution.compact_events(e); r[0]["payload"]["a"].append(1); self.assertEqual(e,old); self.assertNotIn("extra",r[0])
''',
    ),
    Task(
        "data_build_graph",
        "data_infrastructure",
        COMMON + """

Implement `build_order(dependencies)`. The input maps a target name to an iterable of its direct
prerequisite target names; names must be non-empty strings. Prerequisites need not be keys and
must still appear in the result. Return a list containing every target exactly once such that each
prerequisite precedes its dependents. Whenever multiple targets are ready, choose lexicographically.
Reject strings used as dependency iterables and raise ValueError for malformed names or cycles.
Do not mutate the input.
""",
        r'''
import copy, unittest
import solution

class Tests(unittest.TestCase):
    def test_order(self): self.assertEqual(solution.build_order({"app":["db","api"],"api":["core"],"db":["core"]}),["core","api","db","app"])
    def test_implicit(self): self.assertEqual(solution.build_order({"b":["a"]}),["a","b"])
    def test_lexical(self): self.assertEqual(solution.build_order({"z":[],"a":[],"m":[]}),["a","m","z"])
    def test_cycle(self):
        with self.assertRaises(ValueError): solution.build_order({"a":["b"],"b":["a"]})
    def test_bad(self):
        for x in [{"a":"bc"},{"":[]},{"a":[""]}]:
            with self.assertRaises(ValueError): solution.build_order(x)
    def test_no_mutation(self):
        x={"b":["a"]}; old=copy.deepcopy(x); solution.build_order(x); self.assertEqual(x,old)
''',
    ),
    Task(
        "data_config_merge",
        "data_infrastructure",
        COMMON + """

Implement `merge_config(base, override)`. Both inputs must be dictionaries. Return a deep-copied
merge without mutating either input. For a key present in override: `None` deletes it; two dictionary
values merge recursively; otherwise the override value replaces the base value (lists are replaced,
not concatenated). A deletion of a missing key is a no-op. Keys may be strings only; validate keys
recursively in both inputs and raise ValueError otherwise. Preserve normal dictionary insertion order.
""",
        r'''
import copy, unittest
import solution

class Tests(unittest.TestCase):
    def test_nested(self): self.assertEqual(solution.merge_config({"a":{"x":1,"y":2},"b":[1]},{"a":{"y":3,"z":4},"b":[2]}),{"a":{"x":1,"y":3,"z":4},"b":[2]})
    def test_delete(self): self.assertEqual(solution.merge_config({"a":1,"b":2},{"a":None,"c":None}),{"b":2})
    def test_replace_dict_scalar(self): self.assertEqual(solution.merge_config({"a":{"x":1}},{"a":5}),{"a":5})
    def test_no_mutation_deepcopy(self):
        a={"x":{"l":[1]}}; b={"y":{"l":[2]}}; aa=copy.deepcopy(a); bb=copy.deepcopy(b); r=solution.merge_config(a,b); r["x"]["l"].append(3); self.assertEqual(a,aa); self.assertEqual(b,bb)
    def test_bad_key(self):
        for a,b in [({1:"x"},{}),({"x":{2:3}},{}),({}, {3:"x"})]:
            with self.assertRaises(ValueError): solution.merge_config(a,b)
    def test_type(self):
        with self.assertRaises((TypeError,ValueError)): solution.merge_config([], {})
''',
    ),
    Task(
        "science_grid_path",
        "scientific_computing",
        COMMON + """

Implement `shortest_path(grid, start, goal)`. Grid is a non-empty rectangular sequence of rows;
0 is traversable and any other value is blocked. Start and goal are `(row, column)` integer pairs.
Return a shortest four-neighbor path including both endpoints, or None if unreachable. Reject malformed
grids or out-of-bounds/blocked endpoints with ValueError. When several shortest paths exist, make the
result deterministic by exploring neighbors in up, left, right, down order. Do not mutate inputs.
""",
        r'''
import copy, unittest
import solution

class Tests(unittest.TestCase):
    def test_path(self): self.assertEqual(solution.shortest_path([[0,0,0],[1,1,0],[0,0,0]],(0,0),(2,2)),[(0,0),(0,1),(0,2),(1,2),(2,2)])
    def test_tie(self): self.assertEqual(solution.shortest_path([[0,0],[0,0]],(1,0),(0,1)),[(1,0),(0,0),(0,1)])
    def test_same(self): self.assertEqual(solution.shortest_path([[0]],(0,0),(0,0)),[(0,0)])
    def test_none(self): self.assertIsNone(solution.shortest_path([[0,1],[1,0]],(0,0),(1,1)))
    def test_invalid(self):
        for g,s,t in [([], (0,0),(0,0)),([[0],[0,0]],(0,0),(0,0)),([[1]],(0,0),(0,0)),([[0]],(-1,0),(0,0))]:
            with self.assertRaises(ValueError): solution.shortest_path(g,s,t)
    def test_no_mutation(self):
        g=[[0,0]]; old=copy.deepcopy(g); solution.shortest_path(g,(0,0),(0,1)); self.assertEqual(g,old)
''',
    ),
    Task(
        "science_calibration",
        "scientific_computing",
        COMMON + """

Implement `calibration_metrics(probabilities, labels, bins=10)`. Inputs must have equal non-zero
length; probabilities are finite numbers in [0,1], labels are exactly 0 or 1, and bins is a positive
integer. Use equal-width bins `[i/bins,(i+1)/bins)` with probability 1 included in the final bin.
Return `brier`, `ece`, and `bins`. Brier is the mean squared error. ECE is the sample-weighted
absolute confidence-accuracy gap. Each bin record contains index, count, mean_probability, and
positive_rate; use None for both means in empty bins. Preserve full floating-point precision.
""",
        r'''
import math, unittest
import solution

class Tests(unittest.TestCase):
    def test_values(self):
        r=solution.calibration_metrics([0.1,0.4,0.8,1.0],[0,1,1,1],2)
        self.assertAlmostEqual(r["brier"],0.1025); self.assertAlmostEqual(r["ece"],0.225); self.assertEqual([x["count"] for x in r["bins"]],[2,2])
    def test_boundary(self):
        r=solution.calibration_metrics([0.0,0.5,1.0],[0,1,1],2); self.assertEqual([x["count"] for x in r["bins"]],[1,2])
    def test_empty_bins(self):
        r=solution.calibration_metrics([0.1],[0],3); self.assertIsNone(r["bins"][1]["mean_probability"]); self.assertIsNone(r["bins"][1]["positive_rate"])
    def test_validation(self):
        bad=[([],[],2),([.1],[0,1],2),([1.1],[1],2),([float("nan")],[1],2),([.2],[2],2),([.2],[0],0)]
        for p,l,b in bad:
            with self.assertRaises((ValueError,TypeError)): solution.calibration_metrics(p,l,b)
    def test_label_bool(self):
        r=solution.calibration_metrics([.2,.8],[False,True],2); self.assertAlmostEqual(r["brier"],.04)
''',
    ),
    Task(
        "science_formula",
        "scientific_computing",
        COMMON + """

Implement `parse_formula(formula)` and `molecular_mass(formula, atomic_masses)`. A formula uses
element symbols (uppercase followed by optional lowercase), positive integer counts, and nested
parentheses followed by optional positive counts; whitespace is not allowed. `parse_formula`
returns a symbol-to-count dictionary with symbols in first-seen order. Reject empty strings,
unmatched parentheses, zero counts, stray characters, and empty groups with ValueError.
`molecular_mass` sums counts times the supplied numeric masses and raises KeyError for an unknown
element. Do not mutate the mass mapping.
""",
        r'''
import unittest
import solution

class Tests(unittest.TestCase):
    def test_simple(self): self.assertEqual(solution.parse_formula("H2O"),{"H":2,"O":1})
    def test_nested(self): self.assertEqual(solution.parse_formula("K4(ON(SO3)2)2"),{"K":4,"O":14,"N":2,"S":4})
    def test_order(self): self.assertEqual(list(solution.parse_formula("NaClNa2")),["Na","Cl"])
    def test_mass(self): self.assertAlmostEqual(solution.molecular_mass("H2O",{"H":1.008,"O":15.999}),18.015)
    def test_unknown(self):
        with self.assertRaises(KeyError): solution.molecular_mass("Xe",{})
    def test_invalid(self):
        for f in ["","H0","(H2","H2)","()2","2H","H 2","h2"]:
            with self.assertRaises(ValueError,msg=f): solution.parse_formula(f)
''',
    ),
    Task(
        "science_motifs",
        "scientific_computing",
        COMMON + """

Implement `find_motifs(sequence, motifs)`. Sequence and motif strings are DNA symbols from
`ACGTN`, case-insensitive; normalize to uppercase. N is a wildcard in a motif only (sequence N is
literal and matches motif N, but not A/C/G/T). Motifs is an iterable and duplicate motif strings are
rejected. Return a dictionary in input motif order mapping each normalized motif to all zero-based
start indices, including overlapping matches. Reject an empty sequence, empty motif, invalid symbol,
or a motif longer than the sequence with ValueError. Reject a string passed as the motifs iterable.
""",
        r'''
import unittest
import solution

class Tests(unittest.TestCase):
    def test_overlap(self): self.assertEqual(solution.find_motifs("AAAA",["AA"]),{"AA":[0,1,2]})
    def test_wildcard(self): self.assertEqual(solution.find_motifs("ACGTN",["AN","NN","N"]),{"AN":[0],"NN":[0,1,2,3],"N":[0,1,2,3,4]})
    def test_case_order(self): self.assertEqual(list(solution.find_motifs("acgt",["tg","AC"])),["TG","AC"])
    def test_sequence_n_literal(self): self.assertEqual(solution.find_motifs("NN",["A","N"]),{"A":[],"N":[0,1]})
    def test_invalid(self):
        bad=[("",["A"]),("AX",["A"]),("A",[""]),("A",["AA"]),("A","A"),("AA",["A","a"])]
        for s,m in bad:
            with self.assertRaises((ValueError,TypeError)): solution.find_motifs(s,m)
''',
    ),
]


RUNNER = r'''
import importlib.util, json, pathlib, sys, unittest
root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
spec = importlib.util.spec_from_file_location("hidden_case", pathlib.Path(__file__).with_name("test_case.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
result = unittest.TestResult()
suite.run(result)
payload = {"total": result.testsRun, "failures": len(result.failures), "errors": len(result.errors), "passed": result.testsRun-len(result.failures)-len(result.errors)}
print("ORX_TEST_RESULT=" + json.dumps(payload, sort_keys=True))
for test, trace in result.failures + result.errors:
    print("ORX_TEST_DETAIL=" + json.dumps({"test": str(test), "tail": trace[-500:]}, sort_keys=True))
sys.exit(0 if payload["passed"] == payload["total"] else 1)
'''
