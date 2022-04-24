import numpy as np
from collections import deque
from splitters import split_num, split_cat, cmp_cat
from criteria import CRITERIA

## define lib-level dictionaries
NUM_KIND = "num"
CAT_KIND = "cat"

# Each splitter:
# Receives: (x, y, total_classes, impurity_f)
# Returns: (best_threshold, impurity_split, impurity_left, impurity_right)
SPLITTERS = {
    NUM_KIND: split_num,
    CAT_KIND: split_cat
}

# Each comparer:
# Receives: (value, threshold) where value is an array
# Returns: array of bools
COMPARISONS = {
    NUM_KIND: lambda value, threshold: value > threshold,  # vectorized 1 if right, 0 if left
    CAT_KIND: cmp_cat
}

## Node-building function
def build_node(X, y, effective_features, min_samples, impurity_f, feature_kinds, n_classes, node_cost, rng_seed=None):
    """
    Build either a leaf or splitting node.
    
    Params:
    =======
    X (2-d np.array, dtype=float or int): feature values.
    y (1-d np.array, dtype=int): target values, all values must be integers from 0 to n_classes-1.
    effective_features(int, positive): number of features to use. Will be clipped to the [1, X.shape[1]] range.
    min_samples(int, positive): number of minimum samples for splitting a node. If len(y)<min_samples a node 
        is considered a leaf.
    impurity_f (callable): one of the given options for classification impurity function.
    feature_kinds (list of strings): list indicating which kind of feature is which. Must be of the same length as 
        X.shape[1].
    n_classes(int, positive): number of total classes in the dataset. This is required in case some are absent in given y.
    node_cost(float, positive): this node's data cost function value, already computed in a previous node.
    rng_seed(optional, int, positive): if effective_features < X.shape[1], this seed will be used for subset selection.
    
    Returns:
    =======
    node (dict): dictionary containing all node-related information.
    left_cost(float or None): if it's a splitting node, also returns the left-split data impurity value.
    right_cost(float or None): if it's a splitting node, also returns the right-split data impurity value.
    """
    n_samples, n_features = X.shape
      
    # is this a leaf? 
    # if less than min_samples or only one class
    if n_samples < min_samples or len(np.unique(y))==1:
        return {
            'cost': node_cost,
            'leaf': True,
            'value': np.bincount(y, minlength=n_classes) / len(y)
        }, None, None
    
    # calculate considered features
    # effective_features is clipped in the [1, max_features] range
    effective_features = min(max(1, effective_features), n_features)
    if effective_features == n_features:
        features = range(n_features)
    else:
        features = np.random.default_rng(rng_seed).integers(n_features, size=effective_features)
    

    # pre-calculate totals
    total_classes = np.bincount(y, minlength=n_classes)
    
    # O(1)-space best-threshold search
    best_impurity = np.inf      
    for idx in features:
        kind = feature_kinds[idx]
        threshold, split_impurity, left_impurity, right_impurity = SPLITTERS[kind](X[:,idx], y, total_classes, impurity_f)
        if split_impurity < best_impurity:
            best_impurity = split_impurity
            best_threshold = threshold
            best_feature = idx
            best_left = left_impurity
            best_right = right_impurity
    
    return {
        'cost': node_cost,
        'leaf': False,
        'gain': node_cost - split_impurity,
        'feature': best_feature,
        'kind': feature_kinds[best_feature],
        'value': best_threshold
    }, best_left, best_right


## Tree Class
class ClassifierTree:
    ROOT_KEY = '1'
    
    def __init__(self, min_samples, effective_features, criterion, rng_seed = None):
        self.min_samples = min_samples
        self.effective_features = effective_features
        self.rng_seed = np.random.default_rng(rng_seed) if rng_seed is not None else None
        self.impurity_f = CRITERIA[criterion]
    
    def _seed_or_none(self):
        if self.rng_seed is None:
            return None
        return self.rng_seed.integers(20000, size=1)[0]
    
    def fit(self, X, y, kinds):
        # initialize variables
        self.n_classes = len(np.unique(y))
        self.kinds = kinds
        self.nodes = {}
        self.remaining_keys = deque()
        
        
        # the root node is a special case where node_cost must be pre-calculated and all idxs are used
        root_cost = self.impurity_f(np.bincount(y) / len(y))
        valid_idxs = np.full_like(y, True, dtype=np.bool8)
        
        # here we append the (node_key, node_cost, passed_rows) tuple for the node
        self.remaining_keys.append((self.ROOT_KEY, root_cost, valid_idxs))
        
        # while there are remaining keys to be processed
        # process depth-first (as a stack)
        while self.remaining_keys:
            current_key, node_cost, valid_idxs = self.remaining_keys.popleft()
            current_X = X[valid_idxs]
            current_y = y[valid_idxs]
            
            current_node, left_imp, right_imp = build_node(current_X, current_y, 
                                                           effective_features=self.effective_features, 
                                                           min_samples=self.min_samples, 
                                                           impurity_f=self.impurity_f, 
                                                           feature_kinds=kinds, 
                                                           n_classes=self.n_classes,
                                                           node_cost=node_cost, 
                                                           rng_seed=self._seed_or_none())
            # add it to the tree
            self.nodes[current_key] = current_node
            # if it's not a leaf node, should add children
            if not current_node['leaf']:
                splitting_feature = current_node['feature']
                
                # this gives a logical vector with length equal number of samples in train set
                cmp_f = COMPARISONS[kinds[splitting_feature]]
                right_child_valid_idxs = cmp_f(X[:, splitting_feature], current_node['value'])
                
                # append right child to remaining keys
                # key is parent_node_key + 1
                # impurity is right_child_impurity
                # valid idxs is parend_node_valid_idxs & right_child_valid_idxs
                self.remaining_keys.appendleft((current_key+'1', 
                                                right_imp, 
                                                valid_idxs & right_child_valid_idxs))
                
                # now left child
                self.remaining_keys.appendleft((current_key+'0', 
                                                left_imp, 
                                                valid_idxs & ~right_child_valid_idxs))
            
        return self
    
    def predict(self, X):
        if len(X.shape) == 1:
            return self._predict_one(X)
        # assume it's 2 otherwise
        predictions=np.empty(X.shape[0],dtype=int)
        for i in range(X.shape[0]):
            predictions[i] = self._predict_one(X[i])
        return predictions
            
    def _predict_one(self, x):
        current_key = self.ROOT_KEY
        current_node = self.nodes[current_key]
        
        while not current_node['leaf']:
            splitting_feature = current_node['feature']
            cmp_f = COMPARISONS[self.kinds[splitting_feature]]
            split_result = cmp_f(x[splitting_feature], current_node['value'])
            current_key = current_key + str(int(split_result))
            current_node = self.nodes[current_key]
        
        # now it's a leaf node
        return np.argmax(current_node['value'])
    
    def _predict_many(self, X):
        # for predicting many traverse all nodes in parallel instead
        predictions = np.empty(X.shape[0], dtype=int)
        initial_valid_idxs = np.full_like(predictions, True, dtype=np.bool8)
        
        remaining_keys = deque()
        remaining_keys.append((self.ROOT_KEY, initial_valid_idxs))
        
        cnt=0
        while remaining_keys:
            current_key, current_valid_idxs = remaining_keys.popleft()
            current_node = self.nodes[current_key]
            
            cnt+=1
            if cnt % (len(self.nodes) //50) == 0:
                print(f"{cnt}/{len(self.nodes)} nodes traversed")
            # if it's a leaf assign prediction to valid_idxs
            if current_node['leaf']:
                predictions[current_valid_idxs] = np.argmax(current_node['value'])
            else: 
                # if it's a splitting node, update valid_idxs and queue children nodes if needed
                splitting_feature = current_node['feature']
                cmp_f = COMPARISONS[self.kinds[splitting_feature]]
                split_result = cmp_f(X[:,splitting_feature], current_node['value'])
                
                # if at least one node belongs to this path queue the node
                right_child_valid_idxs = current_valid_idxs & split_result
                if sum(right_child_valid_idxs) > 0:
                    remaining_keys.append((current_key+'1', right_child_valid_idxs))
                
                left_child_valid_idxs = current_valid_idxs & ~split_result
                if sum(left_child_valid_idxs) > 0:
                    remaining_keys.append((current_key+'0', left_child_valid_idxs))
        return predictions
    
    def feature_importances(self):
        importances = np.zeros_like(self.kinds, dtype=int)
        # constant-space count (vs. list comprehension + np.bincount)
        for node in self.nodes.values():
            if not node['leaf']:
                importances[node['feature']] +=1
        return importances
