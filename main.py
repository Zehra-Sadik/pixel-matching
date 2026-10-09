import cv2
import numpy as np

from image_loader import load_images
from feature_matching import get_matches
from fundamental_matrix import compute_fundamental_matrix
from epipolar_search import find_match


img1, img2 = load_images()

kp1, kp2, matches = get_matches(
    img1,
    img2
)

F = compute_fundamental_matrix(
    kp1,
    kp2,
    matches
)

left_width = img1.shape[1]


def combine_images(img1, img2):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    max_height = max(h1, h2)

    combined = np.zeros(
        (
            max_height,
            w1 + w2,
            3
        ),
        dtype=np.uint8
    )

    combined[:h1, :w1] = img1
    combined[:h2, w1:w1+w2] = img2

    cv2.putText(
        combined,
        "LEFT IMAGE",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        combined,
        "RIGHT IMAGE",
        (w1 + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    return combined


combined = combine_images(
    img1,
    img2
)


def mouse_callback(
        event,
        x,
        y,
        flags,
        param):

    global combined

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    display = combine_images(
        img1,
        img2
    )

    clicked_left = x < left_width

    if clicked_left:

        line, best_point = find_match(
            img1,
            img2,
            F,
            x,
            y,
            which_image=1
        )

        if line is None or best_point is None:
            return

        a, b, c = line

        rows, cols = img2.shape[:2]

        cv2.drawMarker(
            display,
            (x, y),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            20,
            2
        )

        if abs(b) > 1e-6:

            x0 = 0
            y0 = int(-c / b)

            x1 = cols
            y1 = int(
                -(c + a * cols) / b
            )

            cv2.line(
                display,
                (
                    x0 + left_width,
                    y0
                ),
                (
                    x1 + left_width,
                    y1
                ),
                (0, 255, 0),
                2
            )

        bx, by = best_point

        cv2.drawMarker(
            display,
            (
                bx + left_width,
                by
            ),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
            3
        )

    else:

        right_x = x - left_width

        line, best_point = find_match(
            img2,
            img1,
            F,
            right_x,
            y,
            which_image=2
        )

        if line is None or best_point is None:
            return

        a, b, c = line

        rows, cols = img1.shape[:2]

        cv2.drawMarker(
            display,
            (x, y),
            (255, 0, 0),
            cv2.MARKER_CROSS,
            20,
            2
        )

        if abs(b) > 1e-6:

            x0 = 0
            y0 = int(-c / b)

            x1 = cols
            y1 = int(
                -(c + a * cols) / b
            )

            cv2.line(
                display,
                (x0, y0),
                (x1, y1),
                (0, 255, 0),
                2
            )

        bx, by = best_point

        cv2.drawMarker(
            display,
            (
                bx,
                by
            ),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            20,
            3
        )

    combined = display


cv2.namedWindow(
    "Stereo Matching"
)

cv2.setMouseCallback(
    "Stereo Matching",
    mouse_callback
)

while True:

    cv2.imshow(
        "Stereo Matching",
        combined
    )

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()