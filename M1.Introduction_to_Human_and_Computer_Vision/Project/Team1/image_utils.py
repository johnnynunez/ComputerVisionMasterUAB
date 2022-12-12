import os
from typing import List

import cv2
import numpy as np
from skimage import io

from tqdm import tqdm


def rotate_image(img: np.ndarray, angle=float, mask=False):
    tmpImg = img.copy()
    (h, w) = img.shape[:2]
    origin = (w//2, h//2)
    mat = cv2.getRotationMatrix2D(origin, angle, 1.0)
    rotatedImg = cv2.warpAffine(
        tmpImg, mat, (w, h), flags=cv2.INTER_NEAREST if mask else cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotatedImg


def rotate_images_and_masks_multiple(
        angle_bboxes: List,
        images: List[np.ndarray],
        images_files: List[str],
        masks: List[np.ndarray],
        mssg: str,
        output_dir: str,
):
    for angle_bbox, image, image_file, mask in tqdm(zip(angle_bboxes, images, images_files, masks), total=len(images),
                                                    desc=mssg):
        all_angles = []
        for frame in angle_bbox:
            angle = frame[0]
            all_angles.append(angle)
            
        # print(f"Rotating image: {image_file}, angle_list: {all_angles}")
        rotation_angle = np.mean(all_angles)
        # print(f"Rotating image: {image_file}, angle: {rotation_angle}")
        rotation_angle = -rotation_angle if rotation_angle >= 7 else rotation_angle
        # print(f"Rotating image: {image_file}, angle_after: {rotation_angle}")

        rotated_image = rotate_image(image.copy(), rotation_angle)
        rotated_mask = rotate_image(mask.copy(), rotation_angle, mask=True)

        output_folder = os.path.join(output_dir)
        os.makedirs(output_folder, exist_ok=True)
        # Save images
        io.imsave(
            os.path.join(output_folder, os.path.basename(image_file)),
            (rotated_image).astype(np.uint8),
            check_contrast=False,
        )
        # Save masks
        io.imsave(
            os.path.join(output_folder, os.path.basename(image_file).replace('.jpg', '.png')),
            (rotated_mask).astype(np.uint8),
            check_contrast=False,
        )

    print(f"Saved images to {output_dir}*.jpg")
    print(f"Saved masks to {output_dir}*.png")


def tile_image(img, lvl: int):
    """
    Tiles image into 2^lvl x 2^lvl blocks.
    The block size depends on the lvl and the img size.
    """
    # Estimate block size
    M, N = img.shape[0] // (2 ** (lvl - 1)), img.shape[1] // (2 ** (lvl - 1))
    res_M, res_N = img.shape[0] % lvl, img.shape[1] % lvl
    tiles = [
        img[x:x + M, y:y + N]
        for x in range(0, img.shape[0] - res_M, M)
        for y in range(0, img.shape[1] - res_N, N)
    ]
    return tiles


def tile_image_multilevel(img, n_levels: int):
    """
    Tiles image at multiple levels of resolution.
    """
    tiles_multilevel = []
    for lvl in range(1, n_levels + 1):
        tiles_multilevel.append(tile_image(img, lvl))
    return tiles_multilevel


def find_greatest_contour(image: np.ndarray, num_components: int) -> list:
    contours, _ = cv2.findContours(np.uint8(image), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x1 = 0
    y1 = 0
    w1 = 0
    h1 = 0
    x2 = 0
    y2 = 0
    w2 = 0
    h2 = 0
    area = []

    if num_components == 1:
        for cnt in contours:
            area.append(cv2.contourArea(cnt))
        sorted_list = sorted(area, reverse=True)
        top_n = sorted_list[0:num_components]

        x1, y1, w1, h1 = cv2.boundingRect(contours[area.index(top_n[0])])

        return [x1, y1, w1, h1]

    if num_components == 2:
        for cnt in contours:
            area.append(cv2.contourArea(cnt))
        sorted_list = sorted(area, reverse=True)
        top_n = sorted_list[0:num_components]

        x1, y1, w1, h1 = cv2.boundingRect(contours[area.index(top_n[0])])
        if len(top_n) == 2:
            x2, y2, w2, h2 = cv2.boundingRect(contours[area.index(top_n[1])])

        return [[x1, y1, w1, h1], [x2, y2, w2, h2]]


def crop_image(img, mask):
    x, y, w, h = cv2.boundingRect(mask)
    return (img[y:y + h, x:x + w], (x, y))


def crop_bbox(img: np.ndarray, bbox: List):
    x1, y1, x2, y2 = bbox
    return img[y1:y2, x1:x2]


def crop_paintings(image, mask):
    """
    Crop paintings from an image using the provided mask.
    :param image: image to crop paintings from.
    :param mask: mask to crop paintings from.
    :return: list of cropped paintings.
    """
    import bg_masks
    # get connected components
    masks = bg_masks.fill_connected_component(mask)
    # crop paintings
    paintings = [crop_image(image, m.astype(np.uint8)) for m in masks]
    paintings, origins = zip(*paintings)

    return paintings, origins


def crop_paintings_batch(images, image_filenames, masks, output_dir, extract_text_mask=True, flag_mask=False):
    """
    Crop paintings from a batch of images using the provided masks.
    :param images: batch of images to crop paintings from.
    :param image_filenames: filenames of the images.
    :param masks: batch of masks to crop paintings from.
    :param output_dir: directory to save the cropped paintings.
    :return: list of bounding boxes of the text.
    """
    import detect_textbox

    bboxes_text = []
    paintings_res = []
    mask_text_list = []
    for image, image_filename, mask in zip(images, image_filenames, masks):
        bboxes_query = []
        paintings_query = []
        mask_text_list_aux = []
        paintings, origins = crop_paintings(image, mask)
        image_id = os.path.basename(image_filename).replace(".jpg", "")
        for i, painting in enumerate(paintings):

            output_dir_image = os.path.join(output_dir, image_id)
            os.makedirs(output_dir_image, exist_ok=True)
            io.imsave(
                os.path.join(output_dir_image, str(i) + '.jpg'),
                painting,
                check_contrast=False,
            )

            if extract_text_mask:
                try:
                    bbox_pred = detect_textbox.find_text_bounding_box(painting, kernel_shape=(10, 30))
                except Exception as e:
                    print("*" * 15)
                    print("image_id: ", image_id)
                    print("Error finding text bounding box: {}".format(e))
                    bbox_pred = [1, 1, painting.shape[1] - 1, painting.shape[0] - 1]
                # Snap negative bbox detected points to origin of image (0), if needed.
                bbox_pred = [0 if c < 0 else c for c in bbox_pred]
                bboxes_query.append(
                    [
                        origins[i][0] + bbox_pred[0],
                        origins[i][1] + bbox_pred[1],
                        origins[i][0] + bbox_pred[2],
                        origins[i][1] + bbox_pred[3],
                    ]
                )

                # Save predicted masks in folder.
                mask_text = detect_textbox.create_textbox_mask(painting.shape[0:2], bbox_pred)
                mask_text_list_aux.append(mask_text)
                io.imsave(
                    os.path.join(output_dir_image, str(i) + '.png'),
                    (255 - 255 * mask_text).astype(np.uint8),
                    check_contrast=False
                )
                if flag_mask:
                    painting[1 - mask_text == 1] = 255

            paintings_query.append(painting)

        paintings_res.append(paintings_query)
        bboxes_text.append(bboxes_query)
        mask_text_list.append(mask_text_list_aux)

    print("Done cropping paintings.")
    return bboxes_text, paintings_res, mask_text_list
