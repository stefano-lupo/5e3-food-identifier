# import the necessary packages
import numpy as np
import argparse
import math
import os, glob
import cv2


GRID_SIZE=16


def draw_grid(img, cx, cy, cr, grid_size=GRID_SIZE, use_intersect=True):
    start_x = cx - cr
    start_y = cy - cr
    num_boxes = int(2 * cr / GRID_SIZE)
    for i in range(0, num_boxes):
        x1 = start_x + i * grid_size
        x2 = x1 + grid_size
        for j in range(0, num_boxes):
            y1 = start_y + j * grid_size
            y2 = y1 + grid_size
            print("Checking: (%d, %d) --> (%d, %d)" % (x1, y1, x2, y2))
            if not use_intersect or intersect(cx, cy, cr, x1, y1, grid_size, grid_size):
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 255), 0)


def intersect(cx, cy, cr, rx, ry, rw, rh):
    # return circle_x < rect_x + circle_radius and circle_x + rect_width > rect_x and circle_y < rect_y + circle_radius and rect_height + circle_y > rect_y
    points = [
        (rx, ry),
        (rx + rw, ry),
        (rx, ry + rh),
        (rx + rw, ry + rh)
    ]

    for i, point in enumerate(points):
        if is_point_on_circle(*point, cx, cy, cr):
            # print("point %d on circle (%d, %d)" % (i, point[0], point[1]))
            return True

    return False


def is_point_on_circle(px, py, cx, cy, cr):
    distance = math.sqrt(((cx - px) ** 2) + ((cy - py) ** 2))
    return distance <= cr


def process_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    non_intersect = image.copy()
    intersected = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100, minRadius=100)

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(non_intersect, (x, y), r, (0, 255, 0), 4)
            cv2.circle(intersected, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(non_intersect, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            draw_grid(non_intersect, x, y, r, use_intersect=False)
            draw_grid(intersected, x, y, r, use_intersect=True)


        # show the output image
        cv2.imshow("output", np.hstack([image, non_intersect, intersected]))
        cv2.waitKey(0)


if __name__ == "__main__":
    dir = "../training-files504-all/dataset/train"
    for file in glob.glob(dir + "/*.jpg"):
        process_image(file)