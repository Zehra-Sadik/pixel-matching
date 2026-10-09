import cv2
import numpy as np


def compute_fundamental_matrix(
        kp1,
        kp2,
        matches):

    pts1 = []
    pts2 = []

    for m in matches:

        pts1.append(
            kp1[m.queryIdx].pt
        )

        pts2.append(
            kp2[m.trainIdx].pt
        )

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(
        pts1,
        pts2,
        cv2.FM_RANSAC,
        3,
        0.99
    )

    return F