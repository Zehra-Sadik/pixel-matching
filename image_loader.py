import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename


def load_images():

    root = Tk()
    root.withdraw()

    left_path = askopenfilename(
        title="Select LEFT image"
    )

    right_path = askopenfilename(
        title="Select RIGHT image"
    )

    if not left_path:
        raise Exception("No left image selected")

    if not right_path:
        raise Exception("No right image selected")

    img1 = cv2.imread(left_path)
    img2 = cv2.imread(right_path)

    if img1 is None:
        raise Exception("Failed to load left image")

    if img2 is None:
        raise Exception("Failed to load right image")

    return img1, img2