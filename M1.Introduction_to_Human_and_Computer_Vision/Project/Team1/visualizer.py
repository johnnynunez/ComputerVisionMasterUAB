from skimage.util import img_as_float
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from skimage.transform import resize


def plot_matches_aux(ax, image1, image2, keypoints1, keypoints2, matches,
                     keypoints_color='k', matches_color=None, only_matches=False):
    """Plot matched features.
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Matches and image are drawn in this ax.
    image1 : (N, M [, 3]) array
        First grayscale or color image.
    image2 : (N, M [, 3]) array
        Second grayscale or color image.
    keypoints1 : (K1, 2) array
        First keypoint coordinates as ``(row, col)``.
    keypoints2 : (K2, 2) array
        Second keypoint coordinates as ``(row, col)``.
    matches : (Q, 2) array
        Indices of corresponding matches in first and second set of
        descriptors, where ``matches[:, 0]`` denote the indices in the first
        and ``matches[:, 1]`` the indices in the second set of descriptors.
    keypoints_color : matplotlib color, optional
        Color for keypoint locations.
    matches_color : matplotlib color, optional
        Color for lines which connect keypoint matches. By default the
        color is chosen randomly.
    only_matches : bool, optional
        Whether to only plot matches and not plot the keypoint locations.
    """

    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    new_shape1 = list(image1.shape)
    new_shape2 = list(image2.shape)

    if image1.shape[0] < image2.shape[0]:
        new_shape1[0] = image2.shape[0]
    elif image1.shape[0] > image2.shape[0]:
        new_shape2[0] = image1.shape[0]

    if image1.shape[1] < image2.shape[1]:
        new_shape1[1] = image2.shape[1]
    elif image1.shape[1] > image2.shape[1]:
        new_shape2[1] = image1.shape[1]

    if new_shape1 != image1.shape:
        new_image1 = np.zeros(new_shape1, dtype=image1.dtype)
        new_image1[:image1.shape[0], :image1.shape[1]] = image1
        image1 = new_image1

    if new_shape2 != image2.shape:
        new_image2 = np.zeros(new_shape2, dtype=image2.dtype)
        new_image2[:image2.shape[0], :image2.shape[1]] = image2
        image2 = new_image2

    image = np.concatenate([image1, image2], axis=1)

    offset = image1.shape

    if not only_matches:
        ax.scatter(keypoints1[:, 1], keypoints1[:, 0],
                   facecolors='none', edgecolors=keypoints_color)
        ax.scatter(keypoints2[:, 1] + offset[1], keypoints2[:, 0],
                   facecolors='none', edgecolors=keypoints_color)

    ax.imshow(image, interpolation='nearest', cmap='gray')
    ax.axis((0, 2 * offset[1], offset[0], 0))

    for i in range(matches.shape[0]):
        idx1 = matches[i, 0]
        idx2 = matches[i, 1]

        if matches_color is None:
            color = np.random.rand(3)
        else:
            color = matches_color

        ax.plot((keypoints1[idx1, 1], keypoints2[idx2, 1] + offset[1]),
                (keypoints1[idx1, 0], keypoints2[idx2, 0]),
                '-', color=color)


def plot_matches_(results, qs, db, kp_qs, desc_qs, kp_db, desc_db):
    for idx, i in enumerate(tqdm(results)):
        for jdx, j in enumerate(i):
            if jdx != -1:
                # Query image to grayscale
                im1 = cv2.resize(qs[idx][0], (500, 500))
                im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                im1 = im1.astype(np.uint8)
                kp1, desc1 = kp_qs[idx][0], desc_qs[idx][0]

                # DB image
                im2 = cv2.resize(db[j[0]], (500, 500))
                im2 =  cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
                im2 = im2.astype(np.uint8)
                kp2, desc2 = kp_db[j[0]], desc_db[j[0]]

                index_params = dict(algorithm=0, trees=5)
                search_params = dict(checks=50)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
                # Matching descriptors
                matches = matcher.knnMatch(np.float32(desc1), np.float32(desc2), k=2)
                # Delete possible false positives
                matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
                
                print(len(matches))

                im_matches = cv2.drawMatches(im1, kp1, im2, kp2, matches, None, flags=2)
                plt.figure(figsize = (20,20))
                plt.title(idx)
                plt.imshow(im_matches)
                plt.show()
