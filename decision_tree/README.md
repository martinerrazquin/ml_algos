# Decision Tree Classifier

## Idea

This is just another Decision Tree classifier. Because I didn't want to only write the most basic code I threw some extra things in:

* Out-of-the-box support for categorical features with multiple strategies:
    * Implicit one-hot encoding (one-vs-all) for high cardinality ones
    * Exact matching (best subset) for low cardinality ones
* Multiclass support
* Some degree of optimization without compromising readability (done via Numba)

## Example

A jupyter notebook is provided where the model exhibits the same performance as Scikit-Learn's implementation on the WeatherAUS dataset while not being that much slower.
