# Funk-SVD

## Source

[This](https://sifter.org/~simon/journal/20061211.html) blog post by Simon Funk

## Idea

Given a highly-sparse matrix of ratings R corresponding to *n* users and their ratings on *m* items, we represent each user and item with a vector of *k* latent features, thus forming a *k x n* user-matrix U and a *k x m* item-matrix V, and we propose that R ≈ U.T · V. 

Not only does this simple model perform better than older, non-factorization-based ones but it also does a better job at controlling complexity by making the number of parameters scale with *O(n+m)* instead of *O(n·m)*.

The model is fit via a L2-penalized MSE loss, and can be optimized with either SGD (Stochastic Gradient Descent) or ALS (Alternating Least Squares) methods.

Since this code was originally intended to demonstrate usage of ALS, the method uses no parallelization, hence being super slow (although it's super readable! Code lines follow the equations almost to a tee).

## Example

A jupyter notebook is provided where the model is shown to outperform an user/items-offset-based predictor on the MovieLens-10M dataset, even with randomly-picked hyperparameters.