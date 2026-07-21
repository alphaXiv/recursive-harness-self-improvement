import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Recursive Harness Self-Improvement: bounded reproduction

    **Already-produced evidence:** seven Kubernetes campaigns, eight deterministic tasks,
    56 paired task-run observations, Qwen2.5-Coder-14B-Instruct, and a peak allocation of
    16 NVIDIA RTX PRO 6000 Blackwell GPUs. No expensive inference is run by this notebook.

    The paper asks whether pairwise feedback can revise a prompt-level multi-agent harness so
    that a fixed model beats stronger static test-time scaling. We reconstructed an initial
    harness (H0), two local revisions (H1/H2), and a fixed high-effort control.
    """)
    return


@app.cell
def _():
    evidence = {
        "H0 initial": {"executable": 0.598809, "fixed": 0.658988, "output_tokens": 2096.857, "peak_context": 1980.036},
        "H1 revision": {"executable": 0.661309, "fixed": 0.712113, "output_tokens": 2192.911, "peak_context": 2023.393},
        "H2 revision": {"executable": 0.592262, "fixed": 0.653423, "output_tokens": 2137.839, "peak_context": 1994.518},
        "Static high effort": {"executable": 0.688690, "fixed": 0.735387, "output_tokens": 3519.750, "peak_context": 3124.768},
    }
    return (evidence,)


@app.cell
def _(evidence, mo):
    mo.md(
        "## Headline result\n\n"
        + "| Condition | Executable score | Fixed rubric | Output tokens/task | Peak context |\n"
        + "|---|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {row['executable']:.4f} | {row['fixed']:.4f} | {row['output_tokens']:,.1f} | {row['peak_context']:,.1f} |"
            for name, row in evidence.items()
        )
        + "\n\nH1 improved over H0, but H2 returned to the H0 level and remained below static high effort."
    )
    return


@app.cell
def _(evidence, mo):
    condition = mo.ui.dropdown(options=list(evidence), value="H2 revision", label="Inspect a condition")
    condition
    return (condition,)


@app.cell
def _(condition, evidence, mo):
    selected = evidence[condition.value]
    delta_h0 = selected["executable"] - evidence["H0 initial"]["executable"]
    delta_static = selected["executable"] - evidence["Static high effort"]["executable"]
    mo.callout(
        mo.md(
            f"**{condition.value}**: executable score **{selected['executable']:.4f}**; "
            f"difference from H0 **{delta_h0:+.4f}**; difference from static **{delta_static:+.4f}**."
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Why the judge result is secondary

    Each anonymous pair was judged twice with reversed ordering. Consensus counts were:

    | Comparison | RHI wins | Comparator wins | Ties | Order-consistent |
    |---|---:|---:|---:|---:|
    | H1 vs H0 | 8 | 4 | 44 | 13/56 |
    | H2 vs H0 | 9 | 5 | 42 | 15/56 |
    | H2 vs static | 1 | 12 | 43 | 13/56 |

    Most nominal comparisons became ties because forward and reversed judgments disagreed.
    Hidden executable tests are therefore the primary outcome.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What changed in the harness?

    Natural revisions barely changed the structural counts: H0/H1/H2 averaged
    **3/3/3 roles**, **7/7/7 contract fields**, and **4/4/4.02 hops**. A mechanism
    branch forced H2 to **10.13 contract fields** and **8.5 hops**, but executable
    score fell from **0.7333 to 0.5375** while context grew.

    This bounded result says that adding coordination structure is not sufficient. The content
    of contracts, the stability of revision, and the base model's ability to follow the harness
    are plausible bottlenecks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Assessment

    - **Early local improvement:** partially aligned; H1−H0 was +0.0625, with a wide interval.
    - **Two-revision accumulation:** inconclusive/divergent here; H2−H0 was −0.0065.
    - **Beyond static scaling:** divergent here; H2−static was −0.0964 with a 95% paired
      bootstrap interval of [−0.1702, −0.0238].
    - **Cache efficiency:** not faithfully testable because Transformers exposed no provider-style
      cache read/write counters.

    The notebook is self-contained and intentionally does not rerun the expensive Kubernetes
    generation campaign.
    """)
    return


if __name__ == "__main__":
    app.run()
