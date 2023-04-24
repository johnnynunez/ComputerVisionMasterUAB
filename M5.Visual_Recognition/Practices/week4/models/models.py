import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import resnet18, resnet50
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import boxes as box_ops


import cv2


class EmbeddingNet(nn.Module):
    def __init__(self, weights, resnet_type='resnet50'):
        super(EmbeddingNet, self).__init__()

        if resnet_type == 'resnet50':
            self.resnet = resnet50(weights=weights)
            self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))
            # print dimensionality of the last layer                          
            self.fc = nn.Sequential(nn.Linear(2048, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 2)
                                    )
        elif resnet_type == 'resnet18':
            self.resnet = resnet18(weights=weights)
            self.resnet = nn.Sequential(*(list(self.resnet.children())[:-1]))

            self.fc = nn.Sequential(nn.Linear(512, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 256),
                                    nn.PReLU(),
                                    nn.Linear(256, 2)
                                    )
            

    def forward(self, x):
        output = self.resnet(x).squeeze()
        # output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


def is_target_empty(target):
    if target is None:
        return True

    if all(len(t['boxes']) == 0 and len(t['labels']) == 0 for t in target):
        return True

    return False


# No updates in faster RCNN
# https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
class ObjectEmbeddingNet(nn.Module):
    def __init__(self, weights, resnet_type='V1', weighted = True, with_fc = True):
        super(ObjectEmbeddingNet, self).__init__()
        self.with_fc = with_fc
        self.weighted = weighted

        if resnet_type == 'V1':
            # Load the Faster R-CNN model with ResNet-50 backbone
            self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        elif resnet_type == 'V2':
            self.faster_rcnn = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Replace the box predictor with a custom Fast R-CNN predictor
        in_features = self.faster_rcnn.roi_heads.box_head.fc7.in_features


        # Define the fully connected layers for embedding
        self.fc = nn.Sequential(nn.Sequential(nn.Linear(in_features, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 256),
                                              nn.PReLU(),
                                              nn.Linear(256, 2)
                                              ))

        self.features = []
        def hook_features(module, input, output):
            self.features.append(output)
        self.scores = []
        def hook_scores(module, input, output):
            self.scores.append(output)
        

        layer_to_hook_features = 'roi_heads.box_head.fc7'
        layer_to_hook_scores = 'roi_heads.box_predictor.cls_score'
        for name, layer in self.faster_rcnn.named_modules():
            if name == layer_to_hook_features:
                layer.register_forward_hook(hook_features)
            if name == layer_to_hook_scores:
                layer.register_forward_hook(hook_scores)
   

    def forward(self, x, targets=None):
        targets = {}
        targets['boxes'] = torch.zeros((0,4)).to(x.device)
        targets['labels'] = torch.zeros((0), dtype = torch.int64).to(x.device)
        targets['image_id'] = torch.zeros((0), dtype = torch.int64).to(x.device)
        
        targets = [targets]*x.shape[0]
        
        output = self.faster_rcnn(x, targets)
        
        
        scores = self.scores[0] # 512 boxes per images and 91 scores per box
        self.scores = []
        features = self.features[0] # 512 boxes per images and 1024 features per box
        self.features = []
        
        # Softmax over the scores and get the maximum
        scores_max = nn.functional.softmax(scores, dim=1) # 512 boxes per images and 91 scores per box
        scores_max = torch.max(scores, dim=1)[0] # 512 boxes per images and 1 score per box (the maximum score)
        
        
        if features.shape[0] != 512 * x.shape[0]:  # box_batch_size_per_image = 512  
            print('Number of boxes is not 512')
            # List with the number of boxes per image
            bbox_per_image = self.faster_rcnn.roi_heads.bboxes_per_image
            
            # Split the dim=0 of the features and scores tensors according to the number of boxes per image
            features_img = []
            features_split = []
            scores_split = []
            accumulated_boxes = 0
            for num_boxes in bbox_per_image:
                features_split = features[accumulated_boxes:accumulated_boxes+num_boxes]
                
                if self.weighted:
                    # Obtain weights for the features
                    scores_split = scores_max[accumulated_boxes:accumulated_boxes+num_boxes]
                    scores_split = nn.functional.softmax(scores_split, dim=0).unsqueeze(1)
                    
                    # Weighted features with scores
                    features_weight = torch.mul(features_split, scores_split)
                    
                    features_img.append(torch.sum(features_weight, dim=0))
                else:
                    # Non weighted features
                    features_img.append(torch.mean(features_split, dim=0))
                
                accumulated_boxes += num_boxes
                
                
                
            features_img = torch.stack(features_img, dim=0)
                 
        else:
            # print('Number of boxes is 512')
            features_split = torch.stack(torch.split(features, features.shape[0]//x.shape[0], dim=0),dim=0)
            
            if self.weighted:
                # Obtain weights for the features
                scores_split = torch.stack(torch.split(scores_max, scores_max.shape[0]//x.shape[0], dim=0),dim=0)
                scores_split = nn.functional.softmax(scores_split, dim=1).unsqueeze(2)
                
                # Weighted features with scores
                features_weight = torch.mul(features_split, scores_split)
                
                features_img = torch.sum(features_weight, dim=1)
            else:
                # Non weighted features
                features_img = torch.mean(features_split, dim=1)
        
        if self.with_fc:
            features_img = self.fc(features_img) # [images, 2]
        
        return features_img
    
    
    def get_embedding(self, x):
        return self.forward(x)
    
 

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)



    