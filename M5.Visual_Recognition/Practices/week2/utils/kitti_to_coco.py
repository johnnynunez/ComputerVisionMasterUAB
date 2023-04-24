import json
import os


def load_json_arr(json_path):
    lines = []
    with open(json_path) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines


def get_class_name(class_id):
    classes = {
        0: 'Car',
        1: 'Van',
        2: 'Truck',
        3: 'Pedestrian',
        4: 'Person_sitting',
        5: 'Cyclist',
        6: 'Tram',
        7: 'Misc',
        8: 'DontCare',
    }

    return classes.get(class_id)


def results_kitti(split_path, coco_results_path, kitti_results_path):
    '''
    Car 0.00 0 -1.69 652.16 179.52 699.38 216.18 1.38 1.49 3.32 2.56 1.66 29.10 -1.60
    #Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                         'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                         'Misc' or 'DontCare'
       1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                         truncated refers to the object leaving image boundaries
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
       1    score        Only for results: Float, indicating confidence in
                         detection, needed for p/r curves, higher is better.
    '''
    os.makedirs(kitti_results_path, exist_ok=True)
    split_file = split_path
    dataset_list = []
    with open(split_file) as split_f:
        for line in split_f:
            dataset_list.append(line.strip())

    coco_results_json = load_json_arr(coco_results_path + 'coco_instances_results.json')

    for detection in coco_results_json[0]:
        image_id = detection['image_id']
        image_filename = dataset_list[image_id]

        detection_type = get_class_name(detection['category_id'])
        bbox_left = detection['bbox'][0]
        bbox_top = detection['bbox'][1]
        bbox_width = detection['bbox'][2]
        bbox_height = detection['bbox'][3]
        confidence = detection['score']

        result = "{} -1 -1 -10 {} {} {} {} -1 -1 -1 -1 -1 -1 -1 {}\n".format(
            detection_type, bbox_left, bbox_top, bbox_left + bbox_width, bbox_top + bbox_height, confidence
        )
        f = open(kitti_results_path + image_filename, "a+")
        f.write(result)
        f.close()


if __name__ == "__main__":
    split_path = '/home/mcv/datasets/KITTI/test_kitti.txt'
    coco_results_path = '/ghome/group03/M5-Project/week2/Results/Task_d/faster_RCNN/'
    kitti_results_path = '/ghome/group03/M5-Project/week2/Results/Task_d/faster_RCNN/'

    results_kitti(split_path, coco_results_path, kitti_results_path)
