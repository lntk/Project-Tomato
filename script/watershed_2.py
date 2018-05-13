# ===== SOURCE: https://stackoverflow.com/questions/11294859/how-to-define-the-markers-for-watershed-in-opencv =====

import sys
import cv2
import numpy
from scipy.ndimage import label


def show_image(image, name='image'):
    cv2.imshow(name,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def segment_on_dt(a, img):
    border = cv2.dilate(img, None, iterations=5)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, 2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(numpy.uint8)
    _, dt = cv2.threshold(dt, 180, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now. 
    lbl[border == 255] = 255
    show_image(lbl)

    lbl = lbl.astype(numpy.int32)
    cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(numpy.uint8)
    return 255 - lbl


img = cv2.imread('C:/Users/lengo/Desktop/Project Tomato/image/pears.png')
show_image(img)

# Pre-processing.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    
_, img_bin = cv2.threshold(img_gray, 0, 255,
        cv2.THRESH_OTSU)
img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN,
        numpy.ones((3, 3), dtype=int))

show_image(img_bin)

result = segment_on_dt(img, img_bin)
show_image(result)


result[result != 255] = 0
result = cv2.dilate(result, None)
img[result == 255] = (0, 0, 255)
show_image(img)