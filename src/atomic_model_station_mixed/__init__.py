"""Mixed-convection research primitives; not an accepted station film option."""
from .implicit import FOLD_BO, FOLD_RATIO, RatioResult, solve_ratio
from .laminar import LaminarCase, solve_laminar
from .groups import tube_groups
from .applicability import assess_opposing

__all__ = ['FOLD_BO', 'FOLD_RATIO', 'RatioResult', 'solve_ratio', 'LaminarCase', 'solve_laminar',
           'tube_groups', 'assess_opposing']
