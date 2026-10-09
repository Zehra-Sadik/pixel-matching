import cv2


def get_matches(img1, img2):

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = sift.detectAndCompute(
        gray2,
        None
    )

    bf = cv2.BFMatcher(
        cv2.NORM_L2,
        crossCheck=True
    )

    matches = bf.match(des1, des2)

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )

    good_matches = matches[:100]

    return kp1, kp2, good_matches