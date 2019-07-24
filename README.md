# fast-nnls

This is a Python implementation of the algorithm described in the paper
"A Fast Non-Negativity-Constrained Least Squares Algorithm" by
Rasmus Bro and Sumen De Jong.

Give a matrix `A` and a vector `y`, this algorithm solves `argmin_x ||Ax - y||`.

At the time of writing this, there are no readily available Python bindings
for this algorithm; `scipy.optimize.nnls` implements the slower Lawson-Hanson
version.
