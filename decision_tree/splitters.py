import numpy as np
from numba import njit
from criteria import split_criterion

## define lib-level constants
MAX_EXACT_CARDINALITY = 10

## numeric features
@njit
def get_best_split_num(sorted_x, sorted_y, total_classes, impurity_f):
    """
    Return the best threshold and its split criterion-scores for a given real-valued X data array.
    X is expected to be sorted, and Y must match its order, i.e. (X[i],Y[i]) must belong to the same observation.
    
    Params:
    =======
    sorted_x (1-d np.array, dtype=float or int): feature values, must be sorted.
    sorted_y (1-d np.array, dtype=int): target values, all values must be integers from 0 to len(total_classes)-1.
    total_classes (1-d np.array, dtype=int): total number of occurrences of each class for sorted_y. 
    impurity_f (callable): one of the given options for classification impurity function.
    
    Returns:
    =======
    best_threshold (float): value to be used as threshold.
    best_impurity (float): impurity value of the split.
    best_left (float): impurity value of left split data.
    best_right (float): impurity value of right split data.
    """
    
    best_threshold = None
    best_impurity = np.inf
    
    #initialize count with the first value
    last_x = sorted_x[0]
    current_counts = np.zeros_like(total_classes)
    current_counts[sorted_y[0]] +=1
    
    # current (new unique) x value is assumed to be the first in the right split
    # hence threshold is (current_x + last_x) / 2
    n = total_classes.sum()
    for i in range(1,n):
        current_x = sorted_x[i]
        current_y = sorted_y[i]
        
        # if current_x is not a new unique x value just add it to the bin
        if current_x != last_x:      
            # it it's a new unique x value it's a potential split, measure impurity
            current_impurity, left_impurity, right_impurity = split_criterion(current_counts, total_classes, impurity_f)

            # if it's better than the current best, replace it
            if current_impurity < best_impurity:
                best_impurity = current_impurity
                best_left = left_impurity
                best_right = right_impurity
                best_threshold = (last_x + current_x) / 2

            # that threshold was analyzed, now keep going
            last_x = current_x
        
        current_counts[current_y] +=1
        
    return best_threshold, best_impurity, best_left, best_right

def split_num(x, y, total_classes, impurity_f):
    argsorted_x = np.argsort(x)
    sorted_x = x[argsorted_x]
    sorted_y = y[argsorted_x]
       
    return get_best_split_num(sorted_x, sorted_y, total_classes, impurity_f)


## categorical features
@njit
def get_cat_class_counts(cat_x, k, y, total_classes):
    """
    Return a cat_idx x class_idx -> count matrix.
    E.g., counts[0][1] is the number of ocurrences of class 1 where X is the category 0.
    """
    # count all occurrences for each class
    current_counts = np.zeros((k, len(total_classes)),dtype=np.uint)
    n = total_classes.sum()
    for i in range(n):
        current_x = cat_x[i]
        current_y = y[i]
        current_counts[current_x][current_y] +=1
    return current_counts

@njit
def get_best_split_cat_exact(counts_matrix, k, total_classes, impurity_f):
    best_threshold = None
    best_impurity = np.inf
    
    # iterate from 1 to 2**(k-1):
    # - 0 is not a valid number
    # - always consider the one partition that leaves out last coordinate
    # for readability issues, binary representation is done in inverse order (MSB is the 0th index)
    current_num = np.zeros(k, dtype=np.bool8)
    for _ in range(0,2**k-1):
        # add one - which means finding the first 0-bit, changing it to 1 and resetting the previous bits to 0
        first_zero_bit = np.where(~current_num)[0][0]
        current_num[:first_zero_bit] = 0
        current_num[first_zero_bit] = 1
        
        # if current_num indicates excluded classes, left_threshold is the one where current_num == True
        left_counts = counts_matrix[current_num, :].sum(axis=0)
        
        current_impurity, left_impurity, right_impurity = split_criterion(left_counts, total_classes, impurity_f)
        
        # if it's better than the current best, replace it
        if current_impurity < best_impurity:
            best_impurity = current_impurity
            best_left = left_impurity
            best_right = right_impurity
            best_threshold = ~current_num # save included classes instead 

    return best_threshold, best_impurity, best_left, best_right

@njit
def get_best_split_cat_onehot(counts_matrix, k, total_classes, impurity_f):
    best_threshold = None
    best_impurity = np.inf
    
    # iterate accross possible values for x
    for current_category in range(0, k):
        # use right counts as if they were the left ones, doesn't matter
        right_counts = counts_matrix[current_category, :]
        
        # left is right, right is left
        current_impurity, right_impurity, left_impurity = split_criterion(right_counts, total_classes, impurity_f)
        
        # if it's better than the current best, replace it
        if current_impurity < best_impurity:
            best_impurity = current_impurity
            best_left = left_impurity
            best_right = right_impurity
            best_threshold = current_category

    return best_threshold, best_impurity, best_left, best_right

def split_cat(x, y, total_classes, impurity_f):
    # map unique x values to [[0, k-1]] range
    x = x.astype(int)
    unique_x = np.unique(x).astype(int)
    k = len(unique_x)
    
    mapping = np.empty(unique_x[-1]+1, dtype=int)
    mapping[unique_x] = np.arange(k)
    cat_x = mapping[x]
    
    counts_matrix = get_cat_class_counts(cat_x, k, y, total_classes)
    
    if k > MAX_EXACT_CARDINALITY:
        # use implicit one-hot/one-vs-all method -> threshold is a scalar
        threshold, best_impurity, best_left, best_right = get_best_split_cat_onehot(counts_matrix, k, 
                                                                                    total_classes, impurity_f)
        
        # inverse map on a scalar
        return unique_x[threshold], best_impurity, best_left, best_right
    
    # else use exact method -> threshold is a np.array with bool values
    threshold, best_impurity, best_left, best_right = get_best_split_cat_exact(counts_matrix, k, 
                                                                               total_classes, impurity_f)
    
    # inverse map on a np.array
    return unique_x[np.flatnonzero(threshold)], best_impurity, best_left, best_right

# vectorized comparison for categorical features
def cmp_cat(value, threshold):
    if isinstance(threshold, np.ndarray):
        # if it's a set, return set membership
        return np.in1d(value, threshold)
    else:
        # else its a scalar
        return value == threshold
