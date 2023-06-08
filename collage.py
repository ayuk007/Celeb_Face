import cv2
import numpy as np

def create_collage(img_1, img_2):
    img_1 = cv2.resize(img_1, (400, 350))
    img_2 = cv2.resize(img_2, (400, 350))

    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB)
    collage = np.hstack([img_1, img_2])

    return collage