import numpy as np


def fnnls(A, y, epsilon=1e-5):
    m, n = A.shape
    if y.ndim != 1 or y.shape[0] != m:
        raise ValueError('Invalid dimension; got y vector of size {}, ' \
                         'expected {}'.format(y.shape, m))

    AtA = A.T.dot(A)
    Aty = A.T.dot(y)

    # Represents passive and active sets.
    # If sets[j] is 0, then index j is in the active set (R in literature).
    # Else, it is in the passive set (P).
    sets = np.zeros(n, dtype=np.bool)

    x = np.zeros(n, dtype=np.float64)
    w = Aty
    s = np.zeros(n, dtype=np.float64)

    # While R not empty and max_(n \in R) w_n > epsilon
    while not np.all(sets) and np.max(w[~sets]) > epsilon:
        # Find index of maximum element of w which is in active set.
        j = np.argmax(w[~sets])
        # We have the index in MASKED w.
        # We therefore want the j-th zero in `sets`.
        m = np.where(sets == 0)[0][j]

        # Move index from active set to passive set.
        sets[m] = True

        # Get the rows, cols in AtA corresponding to P
        AtA_in_p = AtA[sets][:, sets]
        # Do the same for Aty
        Aty_in_p = Aty[sets]

        # Update s. Solve (AtA)^p * s^p = (Aty)^p
        s[sets] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
        s[~sets] = 0.

        while np.min(s[sets]) <= 0:
            mask = (s[sets] <= 0)
            alpha = np.min(x[sets][mask] / (x[sets][mask] - s[sets][mask]))
            x += alpha * (s - x)

            # Move all indices j in P such that x[j] = 0 to R
            x_zeros = np.where(x == 0)[0][sets]
            sets[x_zeros] = False

            # Update s. Solve (AtA)^p * s^p = (Aty)^p
            s[sets] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
            s[~sets] = 0.

        x = s
        w = Aty - AtA.dot(x)

    return x
