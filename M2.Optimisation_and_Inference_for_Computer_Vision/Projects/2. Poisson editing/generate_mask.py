import cv2
import numpy as np
import skimage

# Read image
SRC_NAME = 'karim.png'
DST_NAME = 'cars.png'
src = cv2.imread(SRC_NAME)
dst = cv2.imread(DST_NAME)


# resize img to 256x256
def resize_img(img):
    img = cv2.resize(img, (256, 256))
    return img


# src = resize_img(src)
# dst = resize_img(dst)
# cv2.imwrite(SRC_NAME, src)
# cv2.imwrite(DST_NAME, dst)


# select 4 points on the image by interface
# and save them in a list


# draw a square selection and save corner points in a list by img by parameter
def select_square2(img, points, name_window):
    global x_init, y_init

    def mouse_callback(event, x, y, flags, param):
        global x_init, y_init
        if event == cv2.EVENT_LBUTTONDOWN:
            x_init, y_init = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(img, (x_init, y_init), (x, y), (0, 0, 255), 3)
            points.append([x_init, y_init])
            points.append([x, y_init])
            points.append([x, y])
            points.append([x_init, y])
            cv2.imshow(name_window, img)

    cv2.setMouseCallback(name_window, mouse_callback)


name_window = 'Source'
cv2.namedWindow(name_window)
points = []
select_square2(src, points, name_window)

while True:
    cv2.imshow(name_window, src)
    k = cv2.waitKey(300) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        break

# create a mask with the selected points
mask = np.zeros(src.shape, np.uint8)
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(mask, [pts], (255, 255, 255))
# to gray mask
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
cv2.imwrite('mask_src_' + SRC_NAME, mask)

############################################

# get 2D width and high from points list
width = points[1][0] - points[0][0]
high = points[2][1] - points[1][1]


def selection_square(img, points, name_window, width, high):
    global x_init, y_init

    def mouse_callback(event, x, y, flags, param):
        global x_init, y_init
        if event == cv2.EVENT_LBUTTONDOWN:
            x_init, y_init = x, y
            cv2.rectangle(img, (x_init, y_init), (x_init + width, y_init + high), (0, 0, 255), 3)
        elif event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(img, (x_init, y_init), (x_init + width, y_init + high), (0, 0, 255), 3)
            points.append([x_init, y_init])
            points.append([x_init + width, y_init])
            points.append([x_init + width, y_init + high])
            points.append([x_init, y_init + high])
            cv2.imshow(name_window, img)

    cv2.setMouseCallback(name_window, mouse_callback)


name_window = 'Destination'
cv2.namedWindow(name_window)

points = []
selection_square(dst, points, name_window, width, high)

# optimize code below
while True:
    cv2.imshow(name_window, dst)
    k = cv2.waitKey(300) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        break

# create a mask with the selected points
mask = np.zeros(dst.shape, np.uint8)
pts = np.array(points, np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(mask, [pts], (255, 255, 255))
# to gray mask
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
cv2.imwrite('mask_dst_' + DST_NAME, mask)
