import sys
import argparse

# Insert path to the root of the repository
sys.path.insert(0, '/ghome/group03/mcv-m6-2023-team6/week5/vehicle_mtmc')

import pickle as pkl
from collections import defaultdict
import numpy as np
from mot.tracklet import Tracklet
from mtmc.run_mtmc import run_mtmc
import os
from reid.vehicle_reid.load_model import load_model_from_opts
import torch
from reid.feature_extractor import FeatureExtractor
from tools.preprocessing import create_extractor
import cv2
from tqdm import tqdm


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="annotate a video from multicam tracks")
    parser.add_argument("--OF", type=int)
    parser.add_argument("--path", type=str, default = '/ghome/group03/mcv-m6-2023-team6/week5/Results/trackings/MTSC')
    parser.add_argument("--reid_model_opts", type=str, default = '/ghome/group03/mcv-m6-2023-team6/week5/vehicle_mtmc/vehicle_models/resnet50_mixstyle/opts.yaml')
    parser.add_argument("--reid_model_weights", type=str, default = '/ghome/group03/mcv-m6-2023-team6/week5/vehicle_mtmc/vehicle_models/resnet50_mixstyle/net_19.pth')
    args = parser.parse_args()
    
    use_OF = bool(args.OF)
    if use_OF:
        path = f'{args.path}/max_iou_OF'
    else:
        path = f'{args.path}/max_iou'

    ################ GPU #################

    device = torch.device('cuda')

    ################ MODEL #################
    reid_model = load_model_from_opts(args.reid_model_opts,
                                        ckpt=args.reid_model_weights,
                                        remove_classifier=True)
    reid_model.to(device)
    reid_model.eval()
    
    ################ EXTRACTOR #################
    extractor = create_extractor(FeatureExtractor, batch_size=1,
                            model=reid_model)

    ################ COMPUTE THE FEATURES IF THEY DO NOT EXIST #################
    
    for seq in os.listdir(f'{path}/input'):
        # EXTRACTION OF FEATURES FOR EACH TRACKLET
        output_path_features = f'{path}/features/{seq}'
        # Create output directory if it does not exist
        if not os.path.exists(output_path_features):
            os.makedirs(output_path_features)
        
        for c in os.listdir(f'{path}/input/{seq}'):
            
            if c.endswith('.pkl'):
                
                c_name = c.split('.pkl')[0]

                print(c_name)

                tracklets = []

                cam = pkl.load(open(f'{path}/input/{seq}/{c}','rb'))
                
                
                
                
                

                tracklets_sort = defaultdict(list)
                

                for frame_num,data in tqdm(cam.items()):
                    frame = cv2.imread(f"/export/home/group03/dataset/aic19-track1-mtmc-train/train/{seq}/{c_name}/frames/{frame_num}.jpg")
                    for det in data:
                        id = int(det[-1])
                        w = det[3] - det[1]
                        h = det[4] - det[2]

                        box_tlwh = [det[1],det[2],w,h]

                        features = extractor(frame, [box_tlwh])
                        features = torch.tensor(features)

                        tracklets_sort[id].append({'frame':frame_num,'bbox':np.array(box_tlwh),'conf':det[-2],'features':features}) 
                        
                    #if frame_num == len()
                
                print(f'Saving {c_name} features at {output_path_features}')
                with open(f'{output_path_features}/features_{c}','wb') as h:
                    pkl.dump(tracklets_sort,h,protocol = pkl.HIGHEST_PROTOCOL)
                    
    

    # TRACKLETS TO TRACKS
    for seq in os.listdir(f'{path}/features'):
        
        output_path_tracker = f'{path}/mot/{seq}'
        # Create output directory if it does not exist
        if not os.path.exists(output_path_tracker):
            os.makedirs(output_path_tracker)
            
        for c in os.listdir(f'{path}/features/{seq}'):
            tracklets = []

            if c.endswith('.pkl'):
            
                c_name = c.split('.pkl')[0]
            
                tracklets_sort = pkl.load(open(f'{path}/features/{seq}/{c}','rb'))
                for id in tracklets_sort.keys():

                    frames = [track['frame'] for track in tracklets_sort[id]]
                    confs = [track['conf'] for track in tracklets_sort[id]]
                    bboxes = [track['bbox'] for track in tracklets_sort[id]]
                    features = [np.array(track['features'].squeeze(0)) for track in tracklets_sort[id]]
                    
                    tracklet = Tracklet(id)
                    tracklet.frames = frames
                    tracklet.conf = confs
                    tracklet.bboxes = bboxes
                    tracklet.features = features
                    tracklet.compute_mean_feature()

                    tracklets.append(tracklet)

                    with open(f'{output_path_tracker}/mot_{c}','wb') as h:
                        pkl.dump(tracklets,h,protocol=pkl.HIGHEST_PROTOCOL) 
