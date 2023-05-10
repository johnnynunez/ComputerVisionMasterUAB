import os

import cv2
import imageio
import numpy as np
from utils import util

from .metrics import compute_confidences_ap

TOTAL_FRAMES_VIDEO = 2141


def rendering_video(cfg, model, frames_modelling, path_results, ai_gt_path, save=True,
                    display=False):
    model.model_background()

    foreground_gif = []
    foreground_gif_boxes = []

    if not os.path.exists(path_results):
        os.makedirs(path_results)

    det_rects = []
    gt_rects = util.load_from_xml(ai_gt_path)
    # Remove the first "frames_modelling" frames
    gt_rects = {k: v for k, v in gt_rects.items() if
                int(k.split('_')[-1]) >= frames_modelling}

    foreground, I = model.compute_next_foreground(frames_modelling)
    foreground = util.noise_reduction(foreground)
    frame_bbox = util.findBBOX(foreground)

    for frames_id in gt_rects:

        frames_id_num = int(frames_id.split('_')[-1])

        while foreground is not None:

            ret = model.compute_next_foreground(frames_id_num)
            if ret:
                foreground, I = ret
                foreground = util.noise_reduction(foreground)

                if len(model.ch_used) == 2:
                    foreground_gif.append(foreground.mean(axis=-1).astype(np.uint8))  # ADD IMAGE GIF
                else:
                    foreground_gif.append(foreground)

                frame_bbox = util.findBBOX(foreground)

                # if foreground is grayscale, convert to RGB
                if len(foreground.shape) == 2:
                    foreground = cv2.cvtColor(foreground, cv2.COLOR_GRAY2RGB)
                else:
                    if len(model.ch_used) == 2:
                        foreground = cv2.cvtColor(foreground.mean(-1).astype(np.uint8), cv2.COLOR_GRAY2RGB)

                # GT bounding box
                for box in gt_rects[frames_id]:
                    cv2.rectangle(foreground, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                filtered_bboxs = util.filter_boxes(frame_bbox, 1.25, 0.5)
                # Detected bounding box
                for box in filtered_bboxs:
                    if len(box) != 0:
                        cv2.rectangle(foreground, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0),
                                      2)
                    if box == []:
                        continue
                    else:
                        det_rects.append([frames_id, box[0], box[1], box[2], box[3]])

                if len(model.ch_used) == 2:
                    foreground_gif_boxes.append(foreground.mean(axis=-1).astype(np.uint8))  # ADD IMAGE GIF
                else:
                    foreground_gif_boxes.append(foreground)  # ADD IMAGE GIF



            else:
                foreground = None

            if frames_id_num % 100 == 0:
                print(f"{frames_id_num} frames processed...")

            if frames_id_num >= -1:
                break

    print(f"DONE! {frames_id_num} frames processed")
    print(f"Saved to '{path_results}'")

    mAP, mIoU = compute_confidences_ap(gt_rects, len(gt_rects), det_rects)
    print('mAP:', mAP)
    print('mIoU:', mIoU)

    # Save GIF
    if cfg['save']:
        if cfg['run_mode'] == 'Gaussian':
            imageio.mimsave(f'{path_results}/denoised_foreground_alpha{model.alpha}_colorspace{model.colorspace}.gif',
                            foreground_gif[:200])
            imageio.mimsave(f'{path_results}/denoised_foreground_alpha{model.alpha}_colorspace{model.colorspace}.gif',
                            foreground_gif_boxes[:200])
        elif cfg['run_mode'] == 'AdaptiveGaussian':
            imageio.mimsave(f'{path_results}/denoised_foreground_alpha{model.alpha}_rho_{model.p}.gif',
                            foreground_gif[:200])
            imageio.mimsave(f'{path_results}/denoised_foreground_alpha{model.alpha}_rho_{model.p}.gif',
                            foreground_gif_boxes[:200])
        elif cfg['run_mode'] == 'SOTA':
            imageio.mimsave(f'{path_results}/SOTA_{cfg["sota_method"]}.gif', foreground_gif[:200])
            imageio.mimsave(f'{path_results}/SOTA_{cfg["sota_method"]}_boxes.gif', foreground_gif_boxes[:200])

    return mAP, mIoU
