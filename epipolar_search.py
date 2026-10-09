import cv2
import numpy as np
from zncc import zncc


def find_match(
        source_img,
        target_img,
        F,
        x,
        y,
        which_image,
        window_size=5):

    gray_source = cv2.cvtColor(
        source_img,
        cv2.COLOR_BGR2GRAY
    )

    gray_target = cv2.cvtColor(
        target_img,
        cv2.COLOR_BGR2GRAY
    )

    if (
        x-window_size < 0 or
        y-window_size < 0 or
        x+window_size >= gray_source.shape[1] or
        y+window_size >= gray_source.shape[0]
    ):
        return None, None

    point = np.array(
        [[[x, y]]],
        dtype=np.float32
    )

    lines = cv2.computeCorrespondEpilines(
        point,
        which_image,
        F
    )

    a, b, c = lines[0][0]

    patch1 = gray_source[
        y-window_size:y+window_size+1,
        x-window_size:x+window_size+1
    ]

    rows, cols = gray_target.shape

    best_score = -999
    best_point = None

    search_range = 150

    start_x = max(
        0,
        x - search_range
    )

    end_x = min(
        cols,
        x + search_range
    )

    for x2 in range(start_x, end_x):

        if abs(b) < 1e-6:
            continue

        y2 = int(
            -(a*x2 + c)/b
        )

        if (
            x2-window_size < 0 or
            x2+window_size >= cols or
            y2-window_size < 0 or
            y2+window_size >= rows
        ):
            continue

        patch2 = gray_target[
            y2-window_size:y2+window_size+1,
            x2-window_size:x2+window_size+1
        ]

        score = zncc(
            patch1,
            patch2
        )

        if score > best_score:

            best_score = score
            best_point = (
                x2,
                y2
            )

    print("Best Score:", best_score)
    print("Best Point:", best_point)

    return (
        (a, b, c),
        best_point
    )