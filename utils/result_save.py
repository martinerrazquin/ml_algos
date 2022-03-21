import pickle


## Run things once, then save on disk and load it from there afterwards
def load_or_exec(filename, save_f= pickle.dump, load_f= pickle.load):
  """
  Decorator for persisting time-heavy function return values. `save_f` and 
  `load_f` should be the inverse of each other, i.e. x == load_f(save_f(x))

  Parameters
  ----------
  filename: str
      filename as which to save the return value.
  save_f: callable with signature save_f(obj, file), default=Pickle.dump
      function to be called for saving the function return value.
  load_f: callable with signature load_f(file), default=Pickle.load
      function to be called for reading the saved value from disk.

  Usage
  -----
  # Using Pickle
  @load_or_exec("my_filename.pickle")
  def sum_2_nums(a,b):
      return a+b

  # Joblib is sometimes better for Scikit-Learn models
  import joblib
  from sklearn.ensemble import RandomForestClassifier as RFC
  @load_or_exec("rf_default.joblib",save_f=joblib.dump, load_f=joblib.load)
  def train_default_rf():
      return RFC().fit(X_train, y_train)
  """
  def aux(func):
    def inner(*args,**kwargs):
      try:
        # try to just load the saved result
        with open(filename, 'rb') as file_obj:
          return load_f(file_obj)
      except Exception as e:
        print(f"{filename} not found, executing function and then saving")
        # execute actual function and save the result
        ret_val = func(*args,**kwargs)
        with open(filename, 'wb') as file_obj:
          save_f(ret_val, file_obj)
        print(f"Saving result to {filename}")
        return ret_val
    return inner
  return aux
