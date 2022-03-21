import numpy as np
import pandas as pd # implicit usage through label access for R_train and R_valid
from numpy.linalg import inv
import matplotlib.pyplot as plt
from collections import defaultdict


# Alternating Least Squares optimization procedure (one step)
# WARNING: Updates are done INPLACE on U and V
def ALS(U,V,R, O_is, O_js, lda, debug=True):
  n, k = U.shape
  k_, m = V.shape
  assert k==k_, "k doesn't match between U and V"

  # U pass
  if debug:
    print(f"Solving for U")
  for i in range(n):
    o_i = O_is[i]
    r_v = np.zeros(shape=(1,k))
    v_v_t = np.zeros(shape=(k, k))
    for j in o_i:
      v_j = V[:,j].reshape(-1,1)
      
      # r_ij * v_j.T
      r_v += R[(i,j)] * v_j.T

      # v_j @ v_j.T
      v_v_t += v_j @ v_j.T

    U[i,:] = r_v @ inv(v_v_t + lda * np.identity(k))
    
  # V pass
  if debug:
    print(f"Solving for V")
  for j in range(m):
    o_j = O_js[j]
    r_u = np.zeros(shape=(k,1))
    u_t_u = np.zeros(shape=(k, k))
    for i in o_j:
      u_i = U[i,:].reshape(1,-1)
      
      # u_T * r_ij
      r_u += u_i.T * R[(i,j)]

      # u_T @ u
      u_t_u += u_i.T @ u_i

    # NumPy gets angry when we try to assign column vectors, so we need to flatten
    V[:,j] = ( 
        inv(u_t_u + lda * np.identity(k)) @ r_u 
              ).flatten()


# Actual Funk-SVD Matrix Factorization model implementation
# WARNING: R_train and R_valid are assumed to be Pandas Dataframes, 
# with columns "user","item" and "rating" in that exact order
class FunkSVD:
  def __init__(self, k, lda):
    assert type(k)==int, k > 0
    assert lda > 0
    self.k = k
    self.lda = lda

  def fit(self, R_train, R_valid, rng_seed, min_iters=4, max_iters=10, mse_threshold=0.05, debug=False):
    assert min_iters <= max_iters, "Min iters is higher than max iters"
    assert min_iters >= 2, "Min iters can't be 1"
    # seed the rng
    self.rng = np.random.default_rng(rng_seed)

    # reset the historics
    self.valid_mse_historic = []
    self.train_loss_historic = []

    # build some dicts for easier indexing
    self._init_params(R_train)
    

    # now run at most max_iters iterations of ALS
    for i in range(max_iters):
      print(f"Running iteration {i+1}/{max_iters}")

      # one step of ALS, backup U,V
      self.U_last = self.U.copy()
      self.V_last = self.V.copy()

      ALS(self.U, self.V, self.R, 
          self.items_by_user, self.users_by_item, 
          self.lda, debug)

      self.valid_mse_historic.append(self._estimate_MSE(R_valid, debug=debug))
      self.train_loss_historic.append(self._loss(R_train, debug=debug))

      # if MSE hasn't decreased at least mse_threshold from last time, stop

      if len(self.valid_mse_historic) >= max(min_iters,2):
        previous_mse, last_mse = self.valid_mse_historic[-2:]

        # convergence condition, should break
        if last_mse > (1-mse_threshold)*previous_mse:
          # also if error increased, restore previous U, V
          if last_mse > previous_mse:
            self.U = self.U_last
            self.V = self.V_last
          break
    
    if debug and len(self.valid_mse_historic) == max_iters:
      print("Reached max iters without achieving convergence")

  def _estimate_RSS(self, R, debug=True):
    """Estimate Residual Sum of Squares (RSS)"""
    rss = 0.0
    for user_id, item_id, rating in R.itertuples(index=False, name=None):
      # predict value
      y_hat = self.predict(user_id, item_id)

      # add squared residual to total
      rss += (y_hat - rating)**2
    if debug:
      print(f"Estimated RSS is {rss:.3f}")
    return rss

  def _estimate_MSE(self, R, debug=True):
    # MSE = RSS / n
    rows = R.shape[0]
    mse = self._estimate_RSS(R,debug) / rows
    if debug:
      print(f"Estimated MSE is {mse:.3f}")
    return mse

  def _loss(self, R, debug=True):
    # loss = RSS + lambda * (Frobenius(U)^2+Frobenius(V)^2)
    U_norm_sqr = np.sum(self.U **2)
    V_norm_sqr = np.sum(self.V **2)
    loss = self._estimate_RSS(R,debug) + self.lda * (U_norm_sqr + V_norm_sqr)
    if debug:
      print(f"U squared norm is {U_norm_sqr:.3f}, V squared norm is {V_norm_sqr:.3f}, loss is {loss:.3f}")
    return loss

  def _init_params(self, R):
    # if ids range from 0 to max_id, then max_id+1 is the number of unique elements
    n = R["user"].max() + 1
    m = R["item"].max() + 1
    
    # build U, V matrices
    # make them have unitary norm on average
    self.U = self.rng.uniform(low=0,high=1/np.sqrt(self.k),size=(n,self.k))
    self.V = self.rng.uniform(low=0,high=1/np.sqrt(self.k),size=(self.k,m))
    self.U_last = self.U
    self.V_last = self.V
    
    # build the observed R values dict and dicts for O_i and O_j
    self.R = dict()

    self.items_by_user = defaultdict(lambda : [])
    self.users_by_item = defaultdict(lambda : [])
    
    # iterate once through the entire ratings
    for user_id, item_id, rating in R.itertuples(index=False, name=None):
      self.R[(user_id, item_id)] = rating      
      self.items_by_user[user_id].append(item_id)
      self.users_by_item[item_id].append(user_id)

  def predict(self, user_id, item_id):
    # as simple as computing the inner product
    # no need for reshaping in this case
    return self.U[user_id,:] @ self.V[:, item_id]

  def recommend(self, user_id, top_k=10):
    # as simple as computing the product against the whole item matrix
    # sorting and getting the top-k is just added functionality
    estimated_ratings = self.U[user_id, :] @ self.V
    known_items = self.items_by_user[user_id]

    return [item_idx for item_idx in np.argsort(estimated_ratings) 
              if item_idx not in known_items][:-top_k-1:-1]

  def summary(self):
    print("Nº iter\t\tLoss(train)\t\tMSE(valid)")
    for idx, (train, valid) in enumerate(zip(self.train_loss_historic, 
                                             self.valid_mse_historic)):
      print(f"   {idx+1}\t\t{train:.4f}\t\t   {valid:.4f}")

  def plot_metrics(self, log_scale_train=False, log_scale_valid=False):
    fig, ax = plt.subplots(figsize=(10,8))

    values_valid = np.log10(self.valid_mse_historic) if log_scale_valid else self.valid_mse_historic
    values_train = np.log10(self.train_loss_historic) if log_scale_train else self.train_loss_historic

    iters = list(range(1,len(values_valid)+1))
    ax.plot(iters, values_valid, 'o-', color='black', label='valid MSE')
    ax.set_xlabel("Nº iteration")
    ax.set_ylabel("Validation-set MSE", color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_title("Metrics for fitted model (Loss on train, MSE on validation)")

    ax2 = ax.twinx()
    ax2.set_ylabel("Train-set Loss",color='red')
    ax2.plot(iters, values_train, 'o-', color='red', label='train loss')
    ax2.tick_params(axis='y', labelcolor='red')

    return fig, ax, ax2
