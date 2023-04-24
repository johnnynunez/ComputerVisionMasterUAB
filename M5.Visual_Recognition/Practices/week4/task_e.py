import argparse
import json
import ujson
import joblib
import os
import functools 

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_V2_Weights

import wandb

from dataset.triplet_data import TripletCOCODatasetFast as TripletCOCODataset
from models.models import TripletNet
from models.models import ObjectEmbeddingNet
from utils import losses
from utils import trainer
from utils.early_stopper import EarlyStopper



def train(args):   
    wandb.init(project="m5-w4", entity="grup7")
    print(wandb.config)
    
    
    args.margin = wandb.config.margin
    
    name = 'task_e' + '_fc_' + str(args.fc) + '_margin_' + str(args.margin)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    
    # ------------------------------- PATHS --------------------------------
    env_path = os.path.dirname(os.path.abspath(__file__))
    # get path of current file
    dataset_path = '/ghome/group03/mcv/datasets/COCO'
    # dataset_path = '../../datasets/COCO'

    output_path = os.path.join(env_path, 'Results/Task_e', name)

    # Create output path if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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

    # train_annot_path = os.path.join(dataset_path, 'instances_train2014.json')
    # val_annot_path = os.path.join(dataset_path, 'instances_val2014.json')

    object_image_dict = json.load(open(os.path.join(dataset_path, 'mcv_image_retrieval_annotations.json')))
    
    try:
        print('Loading train negative image dict')
        path = os.path.join(dataset_path, 'train_dict_negative_img_low.json')
        with open(path, 'r') as f:
            train_negative_image_dict = ujson.load(f)
        print('Done!')
        
        # print('Loading val negative image dict')
        # path = os.path.join(dataset_path, 'val_dict_negative_img_low.json')
        # with open(path, 'r') as f:
        #     val_negative_image_dict = ujson.load(f)
        # print('Done!')
    except:
        train_negative_image_dict = None
        # val_negative_image_dict = None


    
    transform = torch.nn.Sequential(
                FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
                transforms.Resize((256, 256)),
            )


    
    triplet_train_dataset = TripletCOCODataset(None, object_image_dict, train_path, split_name='train',
                                            dict_negative_img=train_negative_image_dict, transform=transform)
    # triplet_test_dataset = TripletCOCODataset(None, object_image_dict, val_path, split_name='val',
    #                                           dict_negative_img=val_negative_image_dict, transform=transform)

    # ------------------------------- DATALOADER --------------------------------

    
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, num_workers=10)

    triplet_test_loader = None
    
    # ------------------------------- MODEL --------------------------------
    margin = args.margin
    
    if args.resnet_type == 'V1':
        weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    elif args.resnet_type == 'V2':
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        
        
    # Pretrained model from torchvision or from checkpoint
    if args.pretrained:
        embedding_net = ObjectEmbeddingNet(weights=weights,
                                        resnet_type=args.resnet_type, weighted = args.weighted, with_fc = args.fc).to(device)

    model = TripletNet(embedding_net).to(device)
    
    # Set all parameters to be trainable
    for param in model.parameters():
        param.requires_grad = True
        
    # Print the number of trainable parameters
    print('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # --------------------------------- TRAINING --------------------------------

    # Loss function
    loss_func = losses.TripletLoss(margin).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Early stoppper
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    # Learning rate scheduler
    # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)
    lr_scheduler = None

    log_interval = 1

    trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_func, optimizer, lr_scheduler, args.num_epochs,
                device, log_interval, output_path, wandb = wandb, name='task_e')




if __name__ == '__main__':
    
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task E')
    parser.add_argument('--resnet_type', type=str, default='V1', help='Resnet version (V1 or V2)')
    parser.add_argument('--weighted', type=bool, default=True, help='Weighted features')
    parser.add_argument('--fc', type=bool, default=False, help='Use fully connected layer')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str,
                        default='/ghome/group03/M5-Project/week4/checkpoints/best_loss_task_a_finetunning.h5',
                        help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--gpu', type=int, default=7, help='GPU device id')
    args = parser.parse_args()
    
    sweep_config = {
        'name': 'task_e_sweep_fc_False_gpu_7',
        'method': 'grid',
        'parameters':{
            'margin': {
                'values': [1, 10, 50]
            }
        }
    }
    
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="m5-w4", entity="grup7")
    
    wandb.agent(sweep_id, function=functools.partial(train, args))
