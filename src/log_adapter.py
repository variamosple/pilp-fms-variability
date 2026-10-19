#!/usr/bin/env python3
"""Logical interpretation adapter for SLIPCOVER and PITA with observation-mask semantics.

Translates simulated FMS event logs into cplint/SLIPCOVER interpretations while
strictly separating negative failure evidence (attempted=1, failed=0) from
unattempted or censored downstream operations (attempted=0 or censored=1).
"""

from pathlib import Path
from typing import Dict, List, Optional


def build_slipcover_program(
    records: List[Dict],
    initial_theory_clauses: Optional[List[str]] = None,
    megaex_bottom: int = 10,
    max_iter: int = 6,
    max_iter_structure: int = 20,
    maxdepth_var: int = 2,
) -> str:
    """Builds a complete SLIPCOVER training program (.pl) from event log records."""
    valid_events = [
        ev for ev in records
        if ev.get("attempted") == 1 and ev.get("censored", 0) == 0 and ev.get("failed") is not None
    ]

    fold_atoms = [f"e{i}" for i in range(1, len(valid_events) + 1)]
    fold_str = ",".join(fold_atoms)

    in_block = "in([])."
    if initial_theory_clauses:
        joined = ",\n".join(f"({c})" for c in initial_theory_clauses)
        in_block = f"in([\n{joined}\n])."

    header = f"""\
:- use_module(library(slipcover)).
:- sc.

:- set_sc(neg_ex, given).
:- set_sc(megaex_bottom, {megaex_bottom}).
:- set_sc(max_iter, {max_iter}).
:- set_sc(max_iter_structure, {max_iter_structure}).
:- set_sc(maxdepth_var, {maxdepth_var}).
:- set_sc(verbosity, 0).

bg([]).
{in_block}

fold(train, [{fold_str}]).

output(failed/2).

input_cw(material/1).
input_cw(geometry/1).
input_cw(finish_grade/1).

determination(failed/2, material/1).
determination(failed/2, geometry/1).
determination(failed/2, finish_grade/1).

modeh(*, failed(-#op, -#st)).
modeb(*, material(-#mat)).
modeb(*, geometry(-#geom)).
modeb(*, finish_grade(-#grade)).
"""

    models_lines = ["\n% --- LOGICAL INTERPRETATIONS WITH OBSERVATION MASK ---"]
    for idx, ev in enumerate(valid_events, start=1):
        op = ev["operation"]
        st = ev["station"]
        models_lines.append(f"begin(model(e{idx})).")
        if ev["failed"] == 1:
            models_lines.append(f"failed({op}, {st}).")
        else:
            models_lines.append(f"neg(failed({op}, {st})).")
        models_lines.append(f"material({ev['material']}).")
        models_lines.append(f"geometry({ev['geometry']}).")
        models_lines.append(f"finish_grade({ev['finish_grade']}).")
        models_lines.append(f"end(model(e{idx})).\n")

    return header + "\n".join(models_lines)


def write_slipcover_file(records: List[Dict], out_path: Path, **kwargs) -> Path:
    """Writes the generated SLIPCOVER Prolog code to disk."""
    content = build_slipcover_program(records, **kwargs)
    out_path.write_text(content, encoding="utf-8")
    return out_path