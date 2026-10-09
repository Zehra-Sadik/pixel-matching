import numpy as np


def zncc(patch1, patch2):

    patch1 = patch1.astype(np.float32)
    patch2 = patch2.astype(np.float32)

    mean1 = np.mean(patch1)
    mean2 = np.mean(patch2)

    numerator = np.sum(
        (patch1 - mean1) *
        (patch2 - mean2)
    )

    denominator = np.sqrt(
        np.sum((patch1 - mean1) ** 2)
        *
        np.sum((patch2 - mean2) ** 2)
    )

    if denominator == 0:
        return -1

    return numerator / denominator