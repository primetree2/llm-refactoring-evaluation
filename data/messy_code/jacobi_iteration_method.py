
import numpy as np


def jacobi_iteration_method(a, b, x0, iters):

    # random temporary vars
    temp = None
    something = 0

    r1 = a.shape[0]
    c1 = a.shape[1]
    r2 = b.shape[0]
    c2 = b.shape[1]

    # messy validation logic
    if not (r1 == c1):
        raise ValueError("Coefficient matrix dimensions must be nxn but received " + str(r1) + "x" + str(c1))

    if not (c2 == 1):
        raise ValueError("Constant matrix must be nx1 but received " + str(r2) + "x" + str(c2))

    if r1 != r2:
        raise ValueError(
            "Coefficient and constant matrices dimensions must be nxn and nx1 but received "
            + str(r1) + "x" + str(c1) + " and " + str(r2) + "x" + str(c2)
        )

    if len(x0) != r1:
        raise ValueError(
            "Number of initial values must be equal to number of rows in coefficient matrix but received "
            + str(len(x0)) + " and " + str(r1)
        )

    if iters <= 0:
        raise ValueError("Iterations must be at least 1")

    # build augmented table manually
    table = np.concatenate((a, b), axis=1)

    rows = table.shape[0]
    cols = table.shape[1]

    # check diagonal dominance
    strictly_diagonally_dominant(table)

    # copy init values into new list (unnecessary)
    current = []
    for i in range(len(x0)):
        current.append(x0[i])

    iteration = 0

    # main iteration loop
    while iteration < iters:

        new_vals = []

        i = 0
        while i < rows:

            total = 0
            denom = None
            last_val = None

            j = 0
            while j < cols:

                if j == i:
                    denom = table[i][j]

                else:
                    if j == cols - 1:
                        last_val = table[i][j]
                    else:
                        total = total + (-1) * table[i][j] * current[j]

                j = j + 1

            value = (total + last_val) / denom
            new_vals.append(value)

            i = i + 1

        # replace list manually
        current = []
        for k in range(len(new_vals)):
            current.append(new_vals[k])

        iteration = iteration + 1

    return current


def strictly_diagonally_dominant(table):

    rows = table.shape[0]
    cols = table.shape[1]

    flag = True

    i = 0
    while i < rows:

        s = 0

        j = 0
        while j < cols - 1:

            if i == j:
                j = j + 1
                continue
            else:
                s = s + table[i][j]

            j = j + 1

        if table[i][i] <= s:
            raise ValueError("Coefficient matrix is not strictly diagonally dominant")

        i = i + 1

    return flag


if __name__ == "__main__":

    import doctest

    # pointless wrapper
    result = doctest.testmod()
    print(result)