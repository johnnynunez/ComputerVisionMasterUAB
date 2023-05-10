import argparse
import json
# import some common libraries
import os

# Obtain the path of the current file
current_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    # --------------------------------- ARGS --------------------------------- #
    parser = argparse.ArgumentParser(description='Task 1_5: Generate bounding boxes')
    parser.add_argument('--task', type=str, default='Task1_5', help='Task to perform')
    parser.add_argument('--network', type=str, default='retinaNet_video',
                        help='Network to use: faster_RCNN or mask_RCNN')
    parser.add_argument('--json_path', type=str,
                        default='/ghome/group03/mcv-m6-2023-team6/week3/Results/sweep_Task1_4/retinaNet/A/0.01/coco_instances_results.json',
                        help='Path to the json file')
    # parser.add_argument('--json_path', type=str, default='/ghome/group03/mcv-m6-2023-team6/week3/Results/sweep_Task1_4/faster_RCNN/A/0.01/coco_instances_results.json', help='Path to the json file')
    parser.add_argument("--save_vis", type=bool, default=False, help="Save visualizations")
    parser.add_argument("--strategy", type=str, default='A', help="A, B_2, B_3, B_4, C_1, C_2, C_3, C_4")
    args = parser.parse_args()

    # --------------------------------- OUTPUT --------------------------------- #

    output_path = os.path.join(current_path, f'Results/{args.task}/{args.network}/{args.strategy}')

    os.makedirs(output_path, exist_ok=True)

    # --------------------------------- RESULTS --------------------------------- #

    with open(args.json_path) as f:
        json_data = json.load(f)

    # --------------------------------- GENERATE BBOX --------------------------------- #
    output_file_path = os.path.join(output_path, f'bbox_{args.network}_{args.strategy}.txt')
    with open(output_file_path, 'w') as f:
        for entry in json_data:
            image_id = entry['image_id']
            category_id = entry['category_id']
            bbox = entry['bbox']
            conf = entry['score']

            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]

            if category_id == 2:
                print(image_id)
                print(category_id)
                print(conf)
                f.write(f'{image_id}, -1, {x}, {y}, {bbox[2]}, {bbox[3]}, {conf}, -1, -1, -1\n')
