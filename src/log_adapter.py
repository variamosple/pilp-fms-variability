#!/usr/bin/env python3
"""Logical interpretation adapter for SLIPCOVER and PITA with observation-mask semantics.

Translates simulated FMS event logs into cplint/SLIPCOVER interpretations while
strictly separating negative failure evidence (attempted=1, failed=0) from
unattempted or censored downstream operations (attempted=0 or censored=1).
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def build_slipcover_program(
    records: List[Dict],
    include_structural_bias: bool = True,
    initial_theory_clauses: List[str] = None,
    megaex_bottom: int = 20,
    max_iter: int = 10,
    max_iter_structure: int = 20,
    maxdepth_var: int = 4,
) -> str:
    """Builds a complete SLIPCOVER training program (.pl) from event log records."""
    runs = defaultdict(list)
    for rec in records:
        runs[rec["run_id"]].append(rec)

    run_ids = list(runs.keys())
    fold_str = ",".join(run_ids)

    in_block = "in([])."
    if initial_theory_clauses:
        joined = ",\n".join(f"({c})" for c in initial_theory_clauses)
        in_block = f"in([\n{joined}\n])."

    header = f"""\
:- use_module(library(slipcover)).
:- sc.

:- set_sc(megaex_bottom, {megaex_bottom}).
:- set_sc(max_iter, {max_iter}).
:- set_sc(max_iter_structure, {max_iter_structure}).
:- set_sc(maxdepth_var, {maxdepth_var}).
:- set_sc(verbosity, 0).

bg([]).
{in_block}

fold(train, [{fold_str}]).

output(failed/1).

input_cw(assigned/2).
input_cw(material/1).
input_cw(geometry/1).
input_cw(finish_grade/1).

determination(failed/1, assigned/2).
determination(failed/1, material/1).
determination(failed/1, geometry/1).
determination(failed/1, finish_grade/1).

% Language bias enforcing station-local causal consistency (+station)
modeh(*, failed(+station)).
modeb(*, assigned(-#op, +station)).
modeb(*, material(-#mat)).
modeb(*, geometry(-#geom)).
modeb(*, finish_grade(-#grade)).
"""

    models_lines = ["\n% --- LOGICAL INTERPRETATIONS WITH OBSERVATION MASK ---"]
    for r_id in run_ids:
        evs = runs[r_id]
        first = evs[0]
        models_lines.append(f"begin(model({r_id})).")
        models_lines.append(f"material({first['material']}).")
        models_lines.append(f"geometry({first['geometry']}).")
        models_lines.append(f"finish_grade({first['finish_grade']}).")

        for ev in evs:
            # Mask semantics (Section 3.4): only attempted and uncensored steps emit evidence
            if ev["attempted"] == 1 and ev["censored"] == 0 and ev["failed"] is not None:
                st = ev["station"]
                op = ev["operation"]
                models_lines.append(f"assigned({op}, {st}).")
                if ev["failed"] == 1:
                    models_lines.append(f"failed({st}).")
                else:
                    models_lines.append(f"neg(failed({st})).")

        models_lines.append(f"end(model({r_id})).\n")

    return header + "\n".join(models_lines)


def write_slipcover_file(records: List[Dict], out_path: Path, **kwargs) -> Path:
    content = build_slipcover_program(records, **kwargs)
    out_path.write_text(content, encoding="utf-8")
    return out_path