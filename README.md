# Recursive Harness Self-Improvement — bounded open-model reproduction

[![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/alphaXiv/recursive-harness-self-improvement/blob/main/notebooks/rhi_reproduction.py)

This repository tests the central claim of [Recursive Harness Self-Improvement (arXiv:2607.15524)](https://arxiv.org/abs/2607.15524): can two trajectory-local, feedback-conditioned prompt-harness revisions improve repository-building quality beyond both an initial harness and a fixed high-effort harness?

Assessment: **partially reproduced under a downscaled open-model setup**. Across 56 paired task-run observations, one revision improved mean executable score from **0.5988 (H0) to 0.6613 (H1)**, but the second ended at **0.5923 (H2)**. H2 was effectively flat against H0 (paired difference **−0.0065**, 95% bootstrap CI **[−0.0946, 0.0851]**) and trailed the static high-effort control at **0.6887** (difference **−0.0964**, CI **[−0.1702, −0.0238]**). The paper reports Sonnet H2 winning **20/30** comparisons against its max-effort baseline; our open judge gave H2 only **1 win vs 12 static-control wins**, with **43/56 ties** after reversed-order consensus.

The reproduction substitutes Qwen2.5-Coder-14B-Instruct for Claude-family agents, eight deterministic standard-library tasks in two domains for 30 open-ended ML tasks in three domains, and one open-model judge for proprietary evaluators. It ran on Kubernetes with NVIDIA RTX PRO 6000 Blackwell GPUs, a peak allocation of **16 GPUs**, and **0.89 hours** of elapsed campaign wall time.

- [Detailed visual report](reports/rhi-reproduction/report.md)
- [Self-contained tutorial notebook](notebooks/rhi_reproduction.py)
- [Machine-readable aggregate](reports/rhi-reproduction/data/summary.json)

## Experiment log

Every formal node inherited the exact run command shown below. `main` is the publication surface and was not launched as an experiment.

| Branch / experiment | Purpose or change | Exact run command | Assessment / outcome | Compute |
|---|---|---|---|---|
| [`main`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/main) | Public report, notebook, and working protocol | Not run as an experiment (publication surface) | Presentation only | — |
| [`orx/baseline-seed-0`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/baseline-seed-0) | Frozen initial protocol | `bash run.sh` | Setup failure before clone: shell expansion error; no scientific evidence | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/runtime-dependency-and-fail-fast-fix`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/runtime-dependency-and-fail-fast-fix) | Working seed-0 protocol with pinned runtime and fail-fast workers | `bash run.sh` | H0/H1/H2/static executable 0.6083/0.6375/0.5875/0.6458 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/replication-seed-1`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/replication-seed-1) | Independent sampling seed 1 | `bash run.sh` | H2−H0 −0.0833; H2−static −0.1000 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/replication-seed-2`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/replication-seed-2) | Independent sampling seed 2 | `bash run.sh` | H2−H0 +0.1417; H2−static −0.0333 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/replication-seed-3`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/replication-seed-3) | Independent sampling seed 3 | `bash run.sh` | H2−H0 −0.0792; H2−static −0.0958 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/replication-seed-4`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/replication-seed-4) | Independent sampling seed 4 | `bash run.sh` | H2−H0 +0.1250; H2−static −0.1375 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/replication-seed-5`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/replication-seed-5) | Independent sampling seed 5 | `bash run.sh` | H2−H0 −0.1000; H2−static −0.1292 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/enforced-contract-hop-revision`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/enforced-contract-hop-revision) | Force every update to add contracts and recall/retest hops | `bash run.sh` | Contracts 7→10.1 and hops 4→8.5, but H2−H0 −0.1958 | Kubernetes, 4× RTX PRO 6000 Blackwell |
| [`orx/artifact-only-optimizer-ablation`](https://github.com/alphaXiv/recursive-harness-self-improvement/tree/orx/artifact-only-optimizer-ablation) | Remove pairwise history while keeping artifact evidence | `bash run.sh` | H2−H0 +0.0708, but H2−static −0.0917; exploratory due sampling | Kubernetes, 4× RTX PRO 6000 Blackwell |

## Run the protocol

Formal evidence was produced through OpenResearch Kubernetes experiments. The fixed command is:

```bash
bash run.sh
```

The Job manifest requests four GPUs and starts four independent model replicas. Generated repositories are temporary; the terminal log records per-task executable outcomes, blinded forward/reverse judgments, token/context usage, harness structures, and a final machine-readable result.

To explore the already-produced evidence locally without rerunning inference:

```bash
marimo edit notebooks/rhi_reproduction.py
# or
marimo run notebooks/rhi_reproduction.py
```

## License

MIT. The benchmark tasks and reproduction code are original clean-room artifacts; no paper code or proprietary data is redistributed.
