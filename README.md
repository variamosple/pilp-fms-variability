# Learning Probabilistic Process-Variability Rules from Production Logs of a Flexible Manufacturing System with PILP

Official reproducibility repository and experimental benchmark for the study:
**"Learning Probabilistic Process-Variability Rules from Production Logs of a Flexible Manufacturing System with Probabilistic Inductive Logic Programming"**.

## Overview

This repository implements an end-to-end methodological and empirical framework connecting **Product–Process–Resource (PPR) variability modeling** with **Probabilistic Inductive Logic Programming (PILP)** via SLIPCOVER on `cplint`.

The codebase is structured around five core components aligned with the manuscript:
1. **Analytical Semantic Audit (`supplementary/audit_legacy.py`):** Exact probabilistic reconstruction, enumeration of station-failure worlds, sequential completion marginals, and Beta–Binomial sensitivity intervals for the reference three-station system.
2. **Parameterized FMS Discrete-Event Simulator (`src/simulator.py`):** Controlled production log generator for a 6-product furniture family (`chair`, `table`, `stool`, `bench`, `armchair`, `desk`) across multiple workstations with known ground-truth failure and duration mechanisms.
3. **PILP Learning & Inference Pipeline (`src/log_adapter.py`, `src/slipcover_pipeline.py`):** Conversion of event logs with observation masks into logical interpretations, structure/parameter learning via SLIPCOVER, and exact probabilistic querying via PITA.
4. **Controlled Comparative Evaluation (`src/baselines.py`, `src/evaluation.py`):** Multi-seed empirical evaluation (H1–H5) against smoothed Beta–Binomial estimators, Bayesian Networks (`pgmpy`), Process Discovery (`pm4py`), and supervised classifiers (`scikit-learn`).
5. **Versioned Variability Export Bundle (`src/exporter.py`, `examples/`):** Semantic-preserving export into Universal Variability Language (`.uvl`), process precedence contracts (`.json`), and event-indexed LPAD rules (`.pl`).

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
│   └── results_schema.csv
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
└── generated/
    ├── audit_results.json
    ├── audit_macros.tex
    └── benchmark_results_summary.csv