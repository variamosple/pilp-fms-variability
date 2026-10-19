% Auto-generated LPAD risk model learned via SLIPCOVER on cplint
% Hard structural constraints remain strictly in variability_model.uvl
:- use_module(library(pita)).
:- pita.
:- begin_lpad.

failed(Station):1.0000 :-
    assigned(cutting, Station),
    geometry(wide).

failed(Station):0.0412 :-
    assigned(cutting, Station),
    finish_grade(standard).

failed(Station):1.0000 :-
    assigned(cutting, Station),
    material(heavy_wood).

:- end_lpad.
