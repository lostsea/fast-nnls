import numpy as np


def fnnls(A, y, epsilon=None):
    if epsilon is None:
        epsilon = np.finfo(np.float64).eps
        
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
    # The set of all possible indices. Construct P, R by using `sets` as a mask
    ind = np.arange(n, dtype=int)
    P = ind[sets]
    R = ind[~sets]

    x = np.zeros(n, dtype=np.float64)
    w = Aty
    s = np.zeros(n, dtype=np.float64)

    # While R not empty and max_(n \in R) w_n > epsilon
    while not np.all(sets) and np.max(w[R]) > epsilon:
        # Find index of maximum element of w which is in active set.
        j = np.argmax(w[R])
        # We have the index in MASKED w.
        # The real index is stored in the j-th position of R.
        m = R[j]

        # Move index from active set to passive set.
        sets[m] = True
        P = ind[sets]
        R = ind[~sets]

        # Get the rows, cols in AtA corresponding to P
        AtA_in_p = AtA[P][:, P]
        # Do the same for Aty
        Aty_in_p = Aty[P]

        # Update s. Solve (AtA)^p * s^p = (Aty)^p
        s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
        s[R] = 0.

        while np.any(s[P] <= epsilon):
            mask = (s[P] <= epsilon)
            alpha = np.min(x[P][mask] / (x[P][mask] - s[P][mask]))
            x += alpha * (s - x)

            # Move all indices j in P such that x[j] = 0 to R
            # First get all indices where x == 0 in the MASKED x
            zero_mask = (x[P] < epsilon)
            # These correspond to indices in P
            zeros = P[zero_mask]
            # Finally, update the passive/active sets.
            sets[zeros] = False
            P = ind[sets]
            R = ind[~sets]

            # Get the rows, cols in AtA corresponding to P
            AtA_in_p = AtA[P][:, P]
            # Do the same for Aty
            Aty_in_p = Aty[P]

            # Update s. Solve (AtA)^p * s^p = (Aty)^p
            s[P] = np.linalg.lstsq(AtA_in_p, Aty_in_p, rcond=None)[0]
            s[R] = 0.

        x = s.copy()
        w = Aty - AtA.dot(x)

    return x
