# Learning Probabilistic Process-Variability Rules from Production Logs of a Flexible Manufacturing System with PILP

Official reproducibility repository and experimental benchmark for the study:
**"Probabilistic Inductive Logic Programming for Learning Process-Variability Rules in Manufacturing Product Lines"**.

## Overview

This repository implements an end-to-end methodological and empirical framework connecting **Product–Process–Resource (PPR) variability modeling** with **Probabilistic Inductive Logic Programming (PILP)** via SLIPCOVER on `cplint`.

The codebase is structured around five core components aligned with the manuscript:
1. **Analytical Semantic Audit (`supplementary/audit_legacy.py`):** Exact probabilistic reconstruction, enumeration of station-failure worlds, sequential completion marginals, and Beta–Binomial sensitivity intervals for the reference three-station system (Appendix A).
2. **Parameterized FMS Discrete-Event Simulator (`src/simulator.py`):** Controlled production log generator for a 6-product furniture family (`chair`, `table`, `stool`, `bench`, `armchair`, `desk`) across multiple workstations with known ground-truth failure and duration mechanisms.
3. **PILP Learning & Inference Pipeline (`src/log_adapter.py`, `src/slipcover_pipeline.py`):** Conversion of event logs with observation masks into logical interpretations, native structure learning via SWI-Prolog `library(slipcover)`, and exact probabilistic querying via `library(pita)`.
4. **Controlled Comparative Evaluation (`src/baselines.py`, `src/evaluation.py`):** Multi-seed empirical evaluation (H1–H5) across 30 seeds against smoothed Beta–Binomial estimators, Bayesian Networks (`pgmpy`), Process Discovery (`pm4py`), and supervised classifiers (`scikit-learn`).
5. **Versioned Variability Export Bundle (`src/exporter.py`, `examples/`):** Semantic-preserving export into Universal Variability Language (`.uvl`), process precedence contracts (`.json`), and event-indexed LPAD rules (`.pl`), including automated satisfiability equivalence verification (Equation 7) and negative test suites.

## Repository Structure

```text
pilp-fms-variability/
├── README.md
├── requirements.txt
├── setup_cplint.sh
├── supplementary/
│   ├── source_parameters.json
│   ├── audit_legacy.py
│   ├── experiment_protocol.json
│   ├── results_schema.csv
│   ├── generate_paper_figures.py
│   ├── generate_paper_figures2.py
│   └── generate_paper_figures3.py
├── src/
│   ├── simulator.py
│   ├── log_adapter.py
│   ├── slipcover_pipeline.py
│   ├── baselines.py
│   ├── evaluation.py
│   └── exporter.py
├── examples/
│   ├── variability_model.uvl
│   ├── process_contract.json
│   ├── learned_risk_model.pl
│   └── export_manifest.json
├── figures/
│   ├── fig1_workflow_architecture.{pdf,png}
│   ├── fig2_empirical_benchmark_results.{pdf,png}
│   ├── fig_event_to_interpretation.{pdf,png}
│   ├── fig_fms_routes.{pdf,png}
│   ├── fig_h3_robustness.{pdf,png}
│   ├── fig_h4_scalability.{pdf,png}
│   └── fig_h5_maintenance.{pdf,png}
└── generated/
    ├── audit_results.json
    ├── audit_macros.tex
    └── benchmark_results_summary.csv

```

## Quickstart & Reproduction

### 1. Environment Setup

Install SWI-Prolog and Python dependencies:

```bash
sudo apt-get install -y swi-prolog-nox
pip install -r requirements.txt

```

### 2. Analytical Audit (Appendix A)

Run the analytical reproduction script to generate `generated/audit_results.json` and LaTeX macros:

```bash
python3 supplementary/audit_legacy.py --root .

```

### 3. Empirical 30-Seed Benchmark (Section 6, Tables 4–7)

Execute the multi-seed benchmark suite across parallel worker processes:

```bash
PYTHONPATH=. python3 src/evaluation.py --root . --workers 4

```

This produces `generated/benchmark_results_summary.csv` and verifies the exported bundle in `examples/` against Equation (7).

### 4. Regenerate Manuscript Figures

Regenerate publication-quality vector (PDF) and raster (PNG) assets in `figures/`:

```bash
# Architecture & Empirical Reference Overview (Figures 1 and 2)
python3 supplementary/generate_paper_figures.py

# Data Lifecycle & FMS Routing Matrix (Sections 4.4 and 5.2)
python3 supplementary/generate_paper_figures2.py

# H3 Robustness, H4 Scalability, and H5 Maintenance (Sections 6.5–6.7)
python3 supplementary/generate_paper_figures3.py

```
