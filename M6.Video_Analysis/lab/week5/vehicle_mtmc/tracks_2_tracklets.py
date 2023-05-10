import pickle as pkl
from collections import defaultdict
import numpy as np
from mot.attributes import STATIC_ATTRIBUTES, DYNAMIC_ATTRIBUTES
from mot.tracklet import Tracklet
import os
from reid.vehicle_reid.load_model import load_model_from_opts
import torch
from reid.feature_extractor import FeatureExtractor
from tools.preprocessing import create_extractor
import cv2
from tqdm import tqdm
import dill


path = '/ghome/group03/mcv-m6-2023-team6/week5/Results/trackings/MTSC/Max_IoU_Old/S01'
device = torch.device('cuda')

# initialize reid model
reid_model = load_model_from_opts('/ghome/group03/mcv-m6-2023-team6/week5/vehicle_mtmc/vehicle_models/resnet50_mixstyle/opts.yaml',
                                    ckpt='/ghome/group03/mcv-m6-2023-team6/week5/vehicle_mtmc/vehicle_models/resnet50_mixstyle/net_19.pth',
                                    remove_classifier=True)


reid_model.to(device)
reid_model.eval()
extractor = create_extractor(FeatureExtractor, batch_size=1,
                            model=reid_model)


# EXTRACTION OF FEATURES FOR EACH TRACKLET
"""for c in os.listdir(path):
    
    if c.endswith('.pkl'):
        
        c_name = c.split('.pkl')[0]

        print(c_name)

        tracklets = []

        cam = pkl.load(open(f'{path}/{c}','rb'))

        tracklets_sort = defaultdict(list)

        for frame_num,data in tqdm(cam.items()):
            frame = cv2.imread(f"/export/home/group03/dataset/aic19-track1-mtmc-train/train/S01/{c_name}/frames/{frame_num}.jpg")
            for det in data:
                id = int(det[-1])
                w = det[3] - det[1]
                h = det[4] - det[2]

                box_tlwh = [det[1],det[2],w,h]

                features = extractor(frame, [box_tlwh])
                features = torch.tensor(features)

                tracklets_sort[id].append({'frame':frame_num,'bbox':np.array(box_tlwh),'conf':det[-2],'features':features}) 
                
            #if frame_num == len()

        with open(f'/ghome/group03/mcv-m6-2023-team6/week5/vehicle_mtmc/features_{c}','wb') as h:
            pkl.dump(tracklets_sort,h,protocol=pkl.HIGHEST_PROTOCOL) """



# TRACKLETS TO TRACKS
for c in os.listdir(path):
        tracklets = []

        if c.endswith('.pkl'):
        
            c_name = c.split('.pkl')[0]
        
            tracklets_sort = pkl.load(open(f'/ghome/group03/mcv-m6-2023-team6/week5/vehicle_mtmc/features_{c}','rb'))
            for id in tracklets_sort.keys():

                frames = [track['frame'] for track in tracklets_sort[id]]
                confs = [track['conf'] for track in tracklets_sort[id]]
                bboxes = [track['bbox'] for track in tracklets_sort[id]]
                features = [np.array(track['features'].squeeze(0)) for track in tracklets_sort[id]]

                ### COMPUTE MEAN FEATURES 
                """ # compute mean features for tracks and delete frame-by-frame re-id features
                    for track in final_tracks:
                        track.compute_mean_feature()
                        track.features = [] """
                
                    
                tracklet = Tracklet(id)
                tracklet.frames = frames
                tracklet.conf = confs
                tracklet.bboxes = bboxes
                tracklet.features = features
                tracklet.compute_mean_feature()

                
                #tracklet.predict_final_static_attributes()
                #tracklet.finalize_speed()
                tracklets.append(tracklet)



            with open(f'/ghome/group03/mcv-m6-2023-team6/week5/Results/trackings/mot_max_iou/S01/mot_{c}','wb') as h:
                pkl.dump(tracklets,h,protocol=pkl.HIGHEST_PROTOCOL) 



