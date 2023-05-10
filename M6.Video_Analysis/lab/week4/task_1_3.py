import copy
import pickle

from tqdm import tqdm

from task_1_2 import *
from utils.util import load_from_txt, discard_overlaps, filter_boxes, iou

from utils.max_iou_tracking import max_iou_tracking_withParked



estimate_flow = {
    'LK': flow_LK,
    'maskflownet': maskflownet,
    'RAFT': flow_raft,
    'liteflownet': flow_liteflownet

}


def track_memory(tracked_objects):
    delete = []
    for idx in tracked_objects:
        if tracked_objects[idx]['memory'] != tracked_objects[idx]['frame']:
            if tracked_objects[idx]['memory'] <= 5:
                delete.append(idx)

    for idx in delete:
        del tracked_objects[idx]


def max_iou_tracking(path, method, frames_path, conf_threshold=0.6, iou_threshold=0.5):
    total_time = 0.0
    total_frames = 0

    det_boxes = load_from_txt(path, threshold=conf_threshold)
    delta_t = 1 / fps

    track_id = 0
    tracked_objects = {}
    memory = 5

    for frame_id in tqdm(det_boxes):
        total_frames += 1
        start_time = time.time()
        # REMOVE OVERLAPPING BOUNDING BOXES
        boxes = det_boxes[frame_id]
        boxes = discard_overlaps(boxes)
        frame_boxes = filter_boxes(boxes)
        # FIRST FRAME, WE INITIALIZE THE OBJECTS ID

        if not tracked_objects:
            for j in range(len(frame_boxes)):
                # We add the tracking object ID at the end of the list  [[frame,x1, y1, x2, y2, conf, track_id]]
                frame_boxes[j].append(track_id)

                tracked_objects[f'{track_id}'] = {
                    'bbox': [frame_boxes[j][1], frame_boxes[j][2], frame_boxes[j][3], frame_boxes[j][4]],
                    'frame': frame_id, 'memory': 0, 'iou': 0}
                track_id += 1
        else:
            # FRAME N+1 WE COMPARE TO OBJECTS IN FRAME N
            current_frame = np.array(Image.open(os.path.join(frames_path, f'{frame_id}.jpg')))
            previous_frame = np.array(Image.open(os.path.join(frames_path, f'{frame_id - 1}.jpg')))

            flow = estimate_flow[method](previous_frame, current_frame, colType=1)

            for data in previous_tracked_objects.items():
                id, boxB = data
                boxB = np.array(boxB['bbox'])

                # Optical flow estimation for each object
                flow_boxB = flow[int(boxB[1]):int(boxB[3]) + 1, int(boxB[0]):int(boxB[2]) + 1]
                flow_boxB = np.mean(flow_boxB, axis=(0, 1))

                displacement = delta_t * flow_boxB

                # UPDATE step: we add to the previous object position the motion estimated (from optical flow estimation)
                new_bbox_B = [boxB[0] + displacement[0],
                              boxB[1] + displacement[1],
                              boxB[2] + displacement[0],
                              boxB[3] + displacement[1]]

                previous_tracked_objects[id]['new_bbox'] = new_bbox_B

            for i in range(len(frame_boxes)):
                frame_boxes[i][0] = frame_id
                best_iou = 0
                track_id_best = 0
                boxA = [frame_boxes[i][1], frame_boxes[i][2], frame_boxes[i][3], frame_boxes[i][4]]

                for data in previous_tracked_objects.items():
                    id, boxB = data
                    iou_score, _ = iou(boxA, boxB['new_bbox'])

                    if iou_score > best_iou and iou_score >= iou_threshold:
                        best_iou = iou_score
                        track_id_best = id

                if track_id_best == 0 and best_iou == 0:
                    frame_boxes[i].append(track_id)
                    tracked_objects[f'{track_id}'] = {'bbox': boxA, 'frame': frame_id, 'memory': 0, 'iou': best_iou}
                    track_id += 1


                else:
                    if tracked_objects[f'{track_id_best}']['frame'] == frame_id:
                        # CHECK IF THERE IS AN OBJECT WITH THE SAME ID IN THAT FRAME AND CHOOSE THE ONE WITH HIGHEST IOU
                        if best_iou > tracked_objects[f'{track_id_best}']['iou']:
                            tracked_objects[f'{track_id}'] = {'bbox': tracked_objects[f'{track_id_best}']['bbox'],
                                                              'frame': frame_id, 'memory': 0, 'iou': best_iou}
                            wrong_id = [i for i, det in enumerate(frame_boxes) if det[-1] == track_id_best][0]
                            frame_boxes[wrong_id][-1] = track_id
                            track_id += 1

                            frame_boxes[i].append(track_id_best)
                            tracked_objects[f'{track_id_best}']['bbox'] = boxA
                            previous_f = tracked_objects[f'{track_id_best}']['frame']

                            # CHECK IF OBJECTS APPEAR CONSECUTIVE
                            if frame_id - previous_f == 1:
                                tracked_objects[f'{track_id_best}']['memory'] = tracked_objects[f'{track_id_best}'][
                                                                                    'memory'] + 1
                            tracked_objects[f'{track_id_best}']['frame'] = frame_id
                            tracked_objects[f'{track_id_best}']['iou'] = best_iou

                        else:
                            frame_boxes[i].append(track_id)
                            tracked_objects[f'{track_id}'] = {'bbox': boxA, 'frame': frame_id, 'memory': 0,
                                                              'iou': best_iou}
                            track_id += 1


                    else:
                        frame_boxes[i].append(track_id_best)
                        tracked_objects[f'{track_id_best}']['bbox'] = boxA
                        previous_f = tracked_objects[f'{track_id_best}']['frame']

                        # CHECK IF OBJECTS APPEAR CONSECUTIVE
                        if frame_id - previous_f == 1:
                            tracked_objects[f'{track_id_best}']['memory'] = tracked_objects[f'{track_id_best}'][
                                                                                'memory'] + 1
                        tracked_objects[f'{track_id_best}']['frame'] = frame_id
                        tracked_objects[f'{track_id_best}']['iou'] = best_iou

        if frame_id == memory:
            track_memory(tracked_objects)
            memory = memory + frame_id

        previous_tracked_objects = copy.deepcopy(tracked_objects)
        cycle_time = time.time() - start_time
        total_time += cycle_time

    print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    return det_boxes


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--detections', type=str,
                        default="/export/home/group03/mcv-m6-2023-team6/week3/Results/Task1_5/faster_RCNN/A/bbox_faster_RCNN_A.txt",
                        help='.txt file with the object detection')

    parser.add_argument('--frames_path', type=str,
                        default="/export/home/group03/dataset/AICity_data/train/S03/c010/frames/",
                        help='path to folder containing the images to estimate the object tracking with optical flow')

    parser.add_argument('--results_path', type=str, default='Results/Task1_3/',
                        help='path to save results')

    args = parser.parse_args()

    # Get the directory of the current file
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Output path for the results
    output_path = os.path.join(current_dir, args.results_path)

    cap = cv2.VideoCapture(current_dir + f'/../../dataset/AICity_data/train/S03/c010/vdo.avi')
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # methods = ['LK', 'maskflownet', 'RAFT', 'liteflownet']
    method = ['iou']

    # perform grid using the multiple combinations of the parameters using product show progress in tqdm
    for method in methods:
        print('.............Object tracking with optical flow for method: {}....................'.format(method))
        output_path_method = os.path.join(output_path, method)

        start = time.time()
        tracking_boxes = max_iou_tracking(args.detections, method, args.frames_path)

        with open(f'{output_path}/tracking_{method}.pkl', 'wb') as h:
            pickle.dump(tracking_boxes, h, protocol=pickle.HIGHEST_PROTOCOL)

        end = time.time()
