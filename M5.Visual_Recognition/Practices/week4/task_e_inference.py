import argparse
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_V2_Weights

# Import tensorboard from pytorch
import wandb

from dataset.triplet_data import TripletCOCODatasetFast, TripletCOCORetrieval
from models.models import TripletNet
from models.models import ObjectEmbeddingNet
from utils import losses
from utils import trainer
from utils import metrics
from utils import retrieval
from utils.early_stopper import EarlyStopper

from sklearn.neighbors import KNeighborsClassifier

import copy


if __name__ == '__main__':
    
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task E')
    parser.add_argument('--resnet_type', type=str, default='V1', help='Resnet version (V1 or V2)')
    parser.add_argument('--weighted', type=bool, default=True, help='Weighted features')
    parser.add_argument('--fc', type=bool, default=False, help='Use fully connected layer')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str,
                        default='/ghome/group03/M5-Project/week4/Results/Task_e/task_e_fc_False_margin_1/task_e_triplet_2.pth',
                        help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id')
    args = parser.parse_args()
    

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    
    # ------------------------------- PATHS --------------------------------
    env_path = os.path.dirname(os.path.abspath(__file__))
    # get path of current file
    dataset_path = '/ghome/group03/mcv/datasets/COCO'
    # dataset_path = '../../datasets/COCO'

    output_path = os.path.dirname(args.weights)


    # -------------------------------- DEVICE --------------------------------
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
        torch.cuda.amp.GradScaler()
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("cpu")
    else:
        print("CPU is available")
        device = torch.device("cpu")

    # ------------------------------- DATASET --------------------------------
    train_path = os.path.join(dataset_path, 'train2014')
    val_path = os.path.join(dataset_path, 'val2014')
    
    train_labels_path = os.path.join(dataset_path, 'instances_train2014.json')
    val_labels_path = os.path.join(dataset_path, 'instances_val2014.json')

    object_image_dict = json.load(open(os.path.join(dataset_path, 'mcv_image_retrieval_annotations.json')))
    
    
    transform = torch.nn.Sequential(
                FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
                transforms.Resize((256, 256)),
            )

    
    db_dataset_train = TripletCOCORetrieval(train_path, object_image_dict, 
                                            transform, "database", train_labels_path)
    db_dataset_val = TripletCOCORetrieval(val_path, object_image_dict, 
                                            transform, "val", val_labels_path)



    # ------------------------------- DATALOADERS --------------------------------
    db_train_loader = DataLoader(db_dataset_train, batch_size=args.batch_size, shuffle=False,
                                    pin_memory=True, num_workers=10)
    db_val_loader = DataLoader(db_dataset_val, batch_size=args.batch_size, shuffle=False,
                                    pin_memory = True, num_workers=10)
    
    
    # ------------------------------- MODEL --------------------------------
    
    if args.resnet_type == 'V1':
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    elif args.resnet_type == 'V2':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        
        
    # Pretrained model from torchvision or from checkpoint
    if args.pretrained:
        embedding_net = ObjectEmbeddingNet(weights=weights,
                                        resnet_type=args.resnet_type, weighted = args.weighted, with_fc = args.fc).to(device)

    model = TripletNet(embedding_net).to(device)
    
    # Load weights from output_path 

    model.load_state_dict(torch.load(args.weights, map_location=device))
    
        
    # Print the number of trainable parameters
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # --------------------------------- INFERENCE --------------------------------

    # Plot emmbedings
    print("Calculating train database embeddings...")
    start = time.time()
    train_embeddings = metrics.extract_embeddings_coco(db_train_loader, model, device)
    end = time.time()
    print("Time to calculate train database embeddings: ", end - start)
    if train_embeddings.shape[1] == 2:
        path = os.path.join(output_path, 'train_embeddings.png')
        metrics.plot_embeddings_coco(train_embeddings, None, None, 'Database Embeddings', path)
    
    print("Calculating val database embeddings...")
    start = time.time()
    val_embeddings = metrics.extract_embeddings_coco(db_val_loader, model, device)
    end = time.time()
    print("Time to calculate val database embeddings: ", end - start)
    if val_embeddings.shape[1] == 2:
        path = os.path.join(output_path, 'val_embeddings.png')
        metrics.plot_embeddings_coco(val_embeddings, None, None, 'Validation Embeddings', path)

    
    # --------------------------------- RETRIEVAL ---------------------------------
    
    train_embeddings = train_embeddings.reshape(train_embeddings.shape[0], -1)
    val_embeddings = val_embeddings.reshape(val_embeddings.shape[0], -1)
    
    knn = KNeighborsClassifier(n_neighbors=5) 
    
    print("Fitting KNN...")
    start = time.time()
    knn.fit(train_embeddings, list(range(train_embeddings.shape[0])))
    end = time.time()
    print("Time to fit KNN: ", end - start)
    

    print("Calculating KNN...")
    start = time.time()
    dis, neighbors = knn.kneighbors(val_embeddings, return_distance=True)
    end = time.time()
    print("Time to calculate KNN: ", end - start)
    
    
    # Compute positive and negative values
    evaluation = metrics.positives_coco(neighbors, db_dataset_train, db_dataset_val)

    
    # --------------------------------- METRICS ---------------------------------
    metrics.calculate_APs_coco(evaluation, output_path)
    
    metrics.plot_PR_binary(evaluation, output_path)
    
    
    # --------------------------------- SHOW EXAMPLES ---------------------------------
    
    query_list = [0, 1, 2, 3, 4]
    
    retrieval.extract_retrieval_examples(db_dataset_val, neighbors, query_list, output_path) 
    

