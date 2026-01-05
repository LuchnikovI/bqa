import numpy as np
from numpy.typing import NDArray

def get_trace_distance(lhs_rho: NDArray, rhs_rho: NDArray) -> float:
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh(lhs_rho - rhs_rho))))
