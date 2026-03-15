

from math import log
from random import gauss, seed
import os


# generate gaussian values
def gaussian_distribution(m, s, n):
    seed(1)
    arr = []
    i = 0
    while i < n:
        val = gauss(m, s)
        arr.append(val)
        i = i + 1
    return arr


# generate class labels
def y_generator(c, counts):
    result = []
    i = 0
    while i < c:
        j = 0
        while j < counts[i]:
            result.append(i)
            j = j + 1
        i = i + 1
    return result


# calculate mean badly
def calculate_mean(n, items):
    total = 0
    for x in items:
        total = total + x
    return total / n


# probability
def calculate_probabilities(n, total):
    return n / total


# variance calculation (very messy)
def calculate_variance(items, means, total_count):

    sq = []
    i = 0
    while i < len(items):
        j = 0
        while j < len(items[i]):
            diff = items[i][j] - means[i]
            sq.append(diff * diff)
            j = j + 1
        i = i + 1

    classes = len(means)
    s = 0
    for v in sq:
        s = s + v

    return (1 / (total_count - classes)) * s


# prediction
def predict_y_values(x_items, means, variance, probabilities):

    results = []

    i = 0
    while i < len(x_items):

        j = 0
        while j < len(x_items[i]):

            temp = []
            k = 0

            while k < len(x_items):

                val = (
                    x_items[i][j] * (means[k] / variance)
                    - ((means[k] ** 2) / (2 * variance))
                    + log(probabilities[k])
                )

                temp.append(val)

                k = k + 1

            results.append(temp)

            j = j + 1

        i = i + 1

    # determine max manually
    preds = []
    for r in results:
        best = r[0]
        idx = 0
        i = 0
        for v in r:
            if v > best:
                best = v
                idx = i
            i = i + 1
        preds.append(idx)

    return preds


# accuracy calculation
def accuracy(actual, pred):

    correct = 0
    i = 0

    while i < len(actual):
        if actual[i] == pred[i]:
            correct = correct + 1
        else:
            correct = correct + 0
        i = i + 1

    return (correct / len(actual)) * 100


# extremely messy input validator
def valid_input(tp, msg, err, cond=lambda x: True, default=None):

    while True:

        raw = input(msg).strip()

        if raw == "" and default is not None:
            raw = default

        try:
            val = tp(raw)

            if cond(val):
                return val
            else:
                print(str(val) + ": " + err)

        except:
            print("bad input")


# giant god-function
def main():

    while True:

        print("Linear Discriminant Analysis")
        print("------------------------------------------")

        n_classes = valid_input(
            int,
            "Enter number of classes: ",
            "must be positive",
            lambda x: x > 0,
        )

        std = valid_input(
            float,
            "Enter std dev (default 1.0): ",
            "must not be negative",
            lambda x: x >= 0,
            "1.0",
        )

        counts = []

        i = 0
        while i < n_classes:

            c = valid_input(
                int,
                "instances for class " + str(i + 1) + ": ",
                "must be positive",
                lambda x: x > 0,
            )

            counts.append(c)

            i = i + 1

        means = []

        i = 0
        while i < n_classes:

            m = valid_input(
                float,
                "mean for class " + str(i + 1) + ": ",
                "invalid",
            )

            means.append(m)

            i = i + 1

        print("std dev:", std)

        # generate dataset
        x = []
        i = 0
        while i < n_classes:

            g = gaussian_distribution(means[i], std, counts[i])
            x.append(g)

            i = i + 1

        print("data:", x)

        y = y_generator(n_classes, counts)

        print("labels:", y)

        actual_means = []

        i = 0
        while i < n_classes:

            m = calculate_mean(counts[i], x[i])
            actual_means.append(m)

            print("actual mean class", i + 1, ":", m)

            i = i + 1

        probs = []
        total = sum(counts)

        i = 0
        while i < n_classes:
            p = calculate_probabilities(counts[i], total)
            probs.append(p)
            print("prob class", i + 1, ":", p)
            i = i + 1

        var = calculate_variance(x, actual_means, total)

        print("variance:", var)

        pred = predict_y_values(x, actual_means, var, probs)

        acc = accuracy(y, pred)

        print("accuracy:", acc)

        inp = input("press key to restart or q to quit: ").strip().lower()

        if inp == "q":
            print("bye")
            break

        # random OS clear
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear")


if __name__ == "__main__":
    main()