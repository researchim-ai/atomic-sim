"""High-order regular-space and stiff reference evolution, separately qualified."""
from .basis import refined_mode
from .evolution import RefinedResult,evolve_refined,endpoint_momentum
__all__=['refined_mode','RefinedResult','evolve_refined','endpoint_momentum']
from .comparison import compare_refined
__all__ += ['compare_refined']
