import numpy as np
from numba import njit

## define lib-level dictionaries
@njit
def split_criterion(left_counts, total_counts, impurity_f):
    # calculate intermediate values
    left_size = np.sum(left_counts)
    total_size = np.sum(total_counts)
    left_frac = left_size / total_size
    
    left_probs = left_counts / left_size
    right_probs = (total_counts - left_counts) / (total_size - left_size)
    
    left_impurity = impurity_f(left_probs)
    right_impurity = impurity_f(right_probs)
    
    return left_frac * left_impurity + (1-left_frac) * right_impurity, left_impurity, right_impurity

@njit
def entropy_f(probs):
    """ x @ log2(x) but it handles the case where x has values of 0"""
    log2_probs = np.log2(probs)
    return -np.nansum(probs * log2_probs)

@njit
def gini_f(probs):
    return 1-np.sum(np.square(probs))

CRITERIA={
        "entropy": entropy_f,
        "gini": gini_f
}
