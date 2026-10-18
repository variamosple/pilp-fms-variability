#!/usr/bin/env python3
"""Native SWI-Prolog SLIPCOVER learning pipeline and PITA LPAD inference engine.

Executes structure learning (induce/2) and parameter estimation (induce_par/2)
directly inside SWI-Prolog via library(slipcover) and evaluates exact marginal
step-failure probabilities via library(pita) over binary decision diagrams (bddem).
"""

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from src.log_adapter import write_slipcover_file
from src.simulator import BASE_STATION_NOISE, PRODUCT_CATALOG, ProductSpec


@dataclass
class LearnedClause:
    station_var: str
    probability: float
    operation: Optional[str]
    condition_attr: Optional[str]
    condition_val: Optional[str]
    raw_clause: str


def _parse_slipcover_output(stdout_text: str, pruning_threshold: float = 0.01) -> List[LearnedClause]:
    """Parses portrayed LPAD clauses from SWI-Prolog SLIPCOVER output and prunes near-zero weights."""
    marker = "=== SLIPCOVER_LEARNED_PROGRAM ==="
    if marker not in stdout_text:
        return []

    block = stdout_text.split(marker, 1)[1].strip()
    raw_entries = [c.strip() for c in block.split("@@CLAUSE_SEP@@") if c.strip()]
    learned: List[LearnedClause] = []

    for raw in raw_entries:
        m_prob = re.search(r"failed\(([^)]+)\)\s*:\s*([0-9.eE+-]+)", raw)
        if not m_prob:
            continue
        st_var = m_prob.group(1).strip()
        prob = float(m_prob.group(2))
        if prob < pruning_threshold:
            continue

        op_match = re.search(r"assigned\(([a-zA-Z0-9_]+)\s*,", raw)
        operation = op_match.group(1) if op_match else None

        cond_attr, cond_val = None, None
        for attr in ("material", "geometry", "finish_grade"):
            m_attr = re.search(rf"{attr}\(([a-zA-Z0-9_]+)\)", raw)
            if m_attr:
                cond_attr = attr
                cond_val = m_attr.group(1)
                break

        learned.append(
            LearnedClause(
                station_var=st_var,
                probability=round(prob, 6),
                operation=operation,
                condition_attr=cond_attr,
                condition_val=cond_val,
                raw_clause=raw,
            )
        )

    return learned


def run_slipcover_learning(
    records: List[Dict],
    mode: str = "structure_learning",
    pruning_threshold: float = 0.01,
    megaex_bottom: int = 15,
    max_iter: int = 10,
    max_iter_structure: int = 15,
    maxdepth_var: int = 3,
) -> Tuple[List[LearnedClause], float]:
    """Executes native SWI-Prolog SLIPCOVER learning (induce/2 or induce_par/2).

    Modes supported:
    - 'structure_learning': Full clause search + EM parameter estimation (induce/2) with empty initial theory.
    - 'parameter_only': Fixed engineer-supplied operational structure (without hidden product-attribute links)
      fitted via EM over BDDs (induce_par/2).
    - 'oracle_structure': Privileged true generating structure fitted via EM over BDDs (induce_par/2).
    """
    t0 = time.perf_counter()

    initial_clauses = None
    induce_pred = "induce([train], P)"

    if mode == "parameter_only":
        # Engineer-designed fixed structure: knows operations and stations, but not withheld attribute interactions
        induce_pred = "induce_par([train], P)"
        initial_clauses = [
            "failed(A):0.10 :- assigned(cutting, A)",
            "failed(A):0.10 :- assigned(leg_drilling, A)",
            "failed(A):0.05 :- assigned(seat_assembly, A)",
            "failed(A):0.05 :- assigned(top_assembly, A)",
            "failed(A):0.05 :- assigned(sanding, A)",
            "failed(A):0.10 :- assigned(varnishing, A)",
        ]
    elif mode == "oracle_structure":
        # Privileged oracle structure: knows exact withheld relational bodies, fits parameters via EM
        induce_pred = "induce_par([train], P)"
        initial_clauses = [
            "failed(A):0.30 :- assigned(cutting, A), material(heavy_wood)",
            "failed(A):0.20 :- assigned(leg_drilling, A), geometry(wide)",
            "failed(A):0.25 :- assigned(varnishing, A), finish_grade(precision)",
            "failed(A):0.03 :- assigned(cutting, A)",
            "failed(A):0.02 :- assigned(leg_drilling, A)",
            "failed(A):0.02 :- assigned(seat_assembly, A)",
            "failed(A):0.02 :- assigned(top_assembly, A)",
            "failed(A):0.02 :- assigned(sanding, A)",
            "failed(A):0.03 :- assigned(varnishing, A)",
        ]

    with tempfile.TemporaryDirectory() as tmpdir:
        pl_path = Path(tmpdir) / "train_job.pl"
        write_slipcover_file(
            records,
            pl_path,
            initial_theory_clauses=initial_clauses,
            megaex_bottom=megaex_bottom,
            max_iter=max_iter,
            max_iter_structure=max_iter_structure,
            maxdepth_var=maxdepth_var,
        )

        goal = (
            f"consult('{pl_path.as_posix()}'), "
            f"{induce_pred}, "
            "writeln('=== SLIPCOVER_LEARNED_PROGRAM ==='), "
            "forall(member(R, P), (portray_clause(R), writeln('@@CLAUSE_SEP@@'))), "
            "halt."
        )

        res = subprocess.run(
            ["swipl", "-q", "-g", goal],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        if res.returncode != 0 and "=== SLIPCOVER_LEARNED_PROGRAM ===" not in res.stdout:
            raise RuntimeError(
                f"SWI-Prolog SLIPCOVER execution failed (code {res.returncode}):\n{res.stderr}"
            )

        clauses = _parse_slipcover_output(res.stdout, pruning_threshold=pruning_threshold)
        runtime_sec = round(time.perf_counter() - t0, 4)
        return clauses, runtime_sec


def query_pita_step_probabilities(
    clauses: List[LearnedClause], products: List[str]
) -> Dict[Tuple[str, str, str], float]:
    """Queries exact marginal step-failure probabilities inside SWI-Prolog using library(pita)."""
    lpad_rules = []
    for c in clauses:
        prob = max(0.0, min(1.0, c.probability))
        if not c.operation:
            continue
        if c.condition_attr and c.condition_val:
            lpad_rules.append(
                f"failed(P, St):{prob:.6f} :- assigned(P, {c.operation}, St), {c.condition_attr}(P, {c.condition_val})."
            )
        else:
            lpad_rules.append(f"failed(P, St):{prob:.6f} :- assigned(P, {c.operation}, St).")

    # Baseline station noise fallback when no learned clause fires for a station
    for st, b_noise in BASE_STATION_NOISE.items():
        lpad_rules.append(f"failed(P, {st}):{b_noise:.6f} :- assigned(P, _, {st}).")

    facts = []
    queries = []
    for prod_name in products:
        spec = PRODUCT_CATALOG[prod_name]
        facts.append(f"material({prod_name}, {spec.material}).")
        facts.append(f"geometry({prod_name}, {spec.geometry}).")
        facts.append(f"finish_grade({prod_name}, {spec.finish_grade}).")
        for op, st in spec.routing:
            facts.append(f"assigned({prod_name}, {op}, {st}).")
            queries.append((prod_name, op, st))

    pita_program = "\n".join(
        [
            ":- use_module(library(pita)).",
            ":- pita.",
            ":- begin_lpad.",
            *lpad_rules,
            *facts,
            ":- end_lpad.",
        ]
    )

    query_goals = [
        f"(prob(failed({p}, {st}), Prob_{p}_{st}), format('PITA_RES|{p}|{op}|{st}|~8f~n', [Prob_{p}_{st}]))"
        for (p, op, st) in queries
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        q_path = Path(tmpdir) / "pita_queries.pl"
        q_path.write_text(pita_program + "\n", encoding="utf-8")
        full_goal = f"consult('{q_path.as_posix()}'), " + ", ".join(query_goals) + ", halt."
        res = subprocess.run(
            ["swipl", "-q", "-g", full_goal],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

    prob_map: Dict[Tuple[str, str, str], float] = {}
    for line in res.stdout.splitlines():
        if line.startswith("PITA_RES|"):
            _, p_name, op_name, st_name, val_str = line.strip().split("|")
            prob_map[(p_name, op_name, st_name)] = round(float(val_str), 6)

    # Ensure every routed step has a valid entry
    for p_name, op_name, st_name in queries:
        if (p_name, op_name, st_name) not in prob_map:
            spec = PRODUCT_CATALOG[p_name]
            prob_map[(p_name, op_name, st_name)] = predict_step_failure_prob(
                clauses, spec, op_name, st_name
            )

    return prob_map


def predict_step_failure_prob(
    clauses: List[LearnedClause], spec: ProductSpec, operation: str, station: str
) -> float:
    """Evaluates exact LPAD noisy-OR failure probability matching PITA independent disjunctions."""
    p_survival = 1.0 - BASE_STATION_NOISE.get(station, 0.02)
    for c in clauses:
        if c.operation == operation:
            if c.condition_attr is None:
                p_survival *= (1.0 - c.probability)
            elif getattr(spec, c.condition_attr) == c.condition_val:
                p_survival *= (1.0 - c.probability)
    return round(1.0 - p_survival, 6)