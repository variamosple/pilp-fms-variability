% Auto-generated LPAD risk model learned via SLIPCOVER on cplint
% Hard structural constraints remain strictly in variability_model.uvl
:- use_module(library(pita)).
:- pita.
:- begin_lpad.

failed(Station):0.3542 :-
    assigned(cutting, Station),
    material(heavy_wood).

failed(Station):0.2231 :-
    assigned(leg_drilling, Station),
    geometry(wide).

failed(Station):0.2960 :-
    assigned(varnishing, Station),
    finish_grade(precision).

failed(Station):0.0167 :-
    assigned(cutting, Station).

failed(Station):0.0278 :-
    assigned(sanding, Station).

failed(Station):0.0117 :-
    assigned(seat_assembly, Station).

failed(Station):0.0816 :-
    assigned(top_assembly, Station).

failed(Station):0.0178 :-
    assigned(varnishing, Station).

:- end_lpad.