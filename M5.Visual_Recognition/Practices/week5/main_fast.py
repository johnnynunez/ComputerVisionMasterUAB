import argparse
import os
import argparse
import functools 
import wandb
import time
import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, FasterRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_V2_Weights

from dataset.database import ImageDatabase, TextDatabase
from dataset.triplet_data import TripletIm2Text, TripletText2Im
from models.models import TripletNetIm2Text, TripletNetText2Img, EmbeddingNetImage, EmbeddingNetText
from utils import losses
from utils import trainer
from utils import test
from utils.early_stopper import EarlyStopper



def train(args):   
    print('Task: ', args.task)
    
    # if args.sweep:
    """print('Wandb sweep...')
    wandb.init()
    
    # IMPORTANT PUT HERE THE NAME OF VARIABLES THAT YOU WANT TO SWEEP
    args.margin = wandb.config.margin
    args.dim_out_fc = wandb.config.dim_out_fc
    args.learning_rate = wandb.config.lr
    args.network_image = wandb.config.network_image
    args.network_text = wandb.config.network_text"""
    # else:
    #     print('No wandb sweep...')
    wandb=None
        
    # -------------------------------- GPU --------------------------------
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

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
        
    # ------------------------------- PATHS --------------------------------
    dataset_path = '/ghome/group03/mcv/datasets/COCO'
    
    train_path = os.path.join(dataset_path, 'train2014')
    val_path = os.path.join(dataset_path, 'val2014')

    train_annot_path = os.path.join(dataset_path, 'captions_train2014.json')
    val_annot_path = os.path.join(dataset_path, 'captions_val2014.json')
    
    if args.network_text == 'FastText':
        train_annot_embed_path = os.path.join(dataset_path, 'encoded_captions_train2014_fasttext.npy')
        val_annot_embed_path = os.path.join(dataset_path, 'encoded_captions_val2014_fasttext.npy')
    elif args.network_text == 'BERT':
        train_annot_embed_path = os.path.join(dataset_path, 'encoded_captions_train2014_bert.npy')
        val_annot_embed_path = os.path.join(dataset_path, 'encoded_captions_val2014_bert.npy')
    else:
        raise ValueError('Network text not valid')
    
    

    transform = torch.nn.Sequential(
                    FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms(),
                    transforms.Resize((256, 256)),
                )
    
    if args.train:
        env_path = os.path.dirname(os.path.abspath(__file__))
        # dataset_path = '../../datasets/COCO'
        
        name_output = f'{args.task}_{args.network_image}_{args.network_text}_dim_out_fc_{args.dim_out_fc}_margin_{args.margin}_lr_{args.learning_rate}'
        output_path = os.path.join(env_path, f'results_new/{args.task}_new', name_output)  
        print('Output path: ', output_path)
        
        print('TRAINING...')
        print('Network image: ', args.network_image)
        print('Network text: ', args.network_text)
        print('Margin: ', args.margin)
        print('Dim out fc: ', args.dim_out_fc)
        print('Learning rate: ', args.learning_rate)
    
        # Create output path if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)  
            
        # ------------------------------- DATASET --------------------------------
        
       

        if args.task == 'task_a':
            triplet_train_dataset = TripletIm2Text(train_annot_path, train_path, train_annot_embed_path, args.network_text, transform=transform)
        elif args.task == 'task_b':
            triplet_train_dataset = TripletText2Im(train_annot_path, train_path, train_annot_embed_path, args.network_text, transform=transform)
        else:
            raise ValueError('Task not valid')
            
            
        image_val_dataset = ImageDatabase(val_annot_path, val_path, num_samples=100, transform=transform)
        text_val_dataset = TextDatabase(val_annot_path, val_path, val_annot_embed_path, args.network_text, num_samples= 100, transform=transform)
    

        # ------------------------------- DATALOADER --------------------------------
        triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=args.batch_size, shuffle=True,
                                        pin_memory=True, num_workers=10)
        triplet_test_loader = None

        image_val_loader = DataLoader(image_val_dataset, batch_size=args.batch_size, shuffle=False,
                                        pin_memory=True, num_workers=10)
        text_val_loader = DataLoader(text_val_dataset, batch_size=args.batch_size, shuffle=False,
                                        pin_memory=True, num_workers=10)
        
    

        # ------------------------------- MODEL --------------------------------
        num_epochs = args.num_epochs
        learning_rate = args.learning_rate
        margin = args.margin
        weights_image = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        weights_text = args.weights_text
            
        # Pretrained model from torchvision or from checkpoint
        embedding_net_image = EmbeddingNetImage(weights=weights_image, network_image = args.network_image, dim_out_fc = args.dim_out_fc).to(device)
        embedding_net_text = EmbeddingNetText(weights=weights_text, network_text= args.network_text, device=device,  dim_out_fc = args.dim_out_fc).to(device)
        
        embedding_net_image.init_weights()
        embedding_net_text.init_weights()

        if args.task == 'task_a':
            model = TripletNetIm2Text(embedding_net_image, embedding_net_text).to(device)
        elif args.task == 'task_b':
            model = TripletNetText2Img(embedding_net_image, embedding_net_text).to(device)
        else:
            raise ValueError('Task not valid')

        # Set all parameters to be trainable
        for param in model.parameters():
            param.requires_grad = True
            
        # Print the number of trainable parameters
        print('Number of trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        # --------------------------------- TRAINING --------------------------------

        # Loss function
        loss_func = losses.TripletLoss(margin).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Early stoppper
        # early_stopper = EarlyStopper(patience=50, min_delta=10)

        # Learning rate scheduler
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        if args.task == 'task_a':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1, last_epoch=-1, verbose=True)    # 3 steps (step=epoch)
        elif args.task == 'task_b':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int((len(triplet_train_loader.dataset)/(20*args.batch_size))*0.35), gamma=0.1, last_epoch=-1, verbose=True) # 400.000 steps (step=iteration)
        else:
            raise ValueError('Task not valid')

        log_interval = 20

        trainer.fit(args, triplet_train_loader, triplet_test_loader, image_val_dataset, image_val_loader, text_val_dataset, text_val_loader, model, loss_func, optimizer, lr_scheduler, num_epochs,
                    device, log_interval, output_path, name=args.task, wandb = wandb)
        
        
        
        
    else:
        #-------------------------------------- LOAD MODEL --------------------------------------
        # Get the directory where the model is saved
        dir_name = os.path.dirname(args.weights_model)
        output_path_test = os.path.join(dir_name, f'test_{args.num_samples_test}_{args.epoch_test}')
        
        weights_image = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        weights_text = args.weights_text
        
        # If args.weights_model contains 'BERT' we use the BERT tokenizer, otherwise we use the default tokenizer
        if args.weights_model.find('BERT') != -1:
            args.network_text = 'BERT'
        elif args.weights_model.find('FastText') != -1:
            args.network_text = 'FastText'
        else:
            raise ValueError('Network TEXT not recognized')
        
        if args.weights_model.find('fasterRCNN') != -1:
            args.network_image = 'fasterRCNN'
        elif args.weights_model.find('RESNET50') != -1:
            args.network_image = 'RESNET50'
        else:
            raise ValueError('Network IMAGE not recognized')
        
        if args.network_text == 'FastText':
            val_annot_embed_path = os.path.join(dataset_path, 'encoded_captions_val2014_fasttext.npy')
        elif args.network_text == 'BERT':
            val_annot_embed_path = os.path.join(dataset_path, 'encoded_captions_val2014_bert.npy')
        else:
            raise ValueError('Network text not valid')
            
        # Pretrained model from torchvision or from checkpoint
        embedding_net_image = EmbeddingNetImage(weights=weights_image, network_image = args.network_image, dim_out_fc = args.dim_out_fc).to(device)
        embedding_net_text = EmbeddingNetText(weights=weights_text, network_text= args.network_text, device=device,  dim_out_fc = args.dim_out_fc).to(device)

        if args.task == 'task_a':
            model = TripletNetIm2Text(embedding_net_image, embedding_net_text).to(device)
        elif args.task == 'task_b':
            model = TripletNetText2Img(embedding_net_image, embedding_net_text).to(device)
        else:
            raise ValueError('Task not recognized')
        
        
        print('LOADING MODEL...')
        model.load_state_dict(torch.load(args.weights_model, map_location=device))
            
        
        
    # --------------------------------- TESTING --------------------------------
    
    num_samples = [100, 500, 1000]
    
    for num in num_samples:
        args.num_samples_test = num
        output_path_test = os.path.join(output_path, f'test_{args.num_samples_test}_{args.epoch_test}')
        
        # Create output path if it does not exist
        if not os.path.exists(output_path_test):
            os.makedirs(output_path_test)
            
        
        image_test_dataset = ImageDatabase(val_annot_path, val_path, num_samples=args.num_samples_test, transform=transform)
        text_test_dataset = TextDatabase(val_annot_path, val_path, val_annot_embed_path, args.network_text, num_samples= args.num_samples_test, transform=transform)
        
        image_test_loader = DataLoader(image_test_dataset, batch_size=args.batch_size, shuffle=False,
                                        pin_memory=True, num_workers=10)
        text_test_loader = DataLoader(text_test_dataset, batch_size=args.batch_size, shuffle=False,
                                        pin_memory=True, num_workers=10)

        map_value = test.test(args, model, image_test_dataset, image_test_loader, text_test_dataset, text_test_loader, output_path_test, device=device, wandb=wandb, retrieval=True)
    
    
    
# Run a sweep in wandb: adapt the sweep_config dictionary below and run:
# python main.py --task_a --train True --sweep True --network_image RESNET50 --network_text FastText --batch_size 64

# Run task_a train example:
# python main.py --task task_a --train True --sweep False --network_image RESNET50 --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
# python main.py --task task_a --train True --sweep False --network_image RESNET101 --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1
# python main.py --task task_a --train True --sweep False --network_image fasterRCNN --network_text FastText --batch_size 64 --lr 1e-4 --margin 0.1

# Run task_a test example:
# python main.py --task task_a --train False --sweep False --network_image RESNET50 --network_text FastText --weights_model /ghome/group03/M5-Project/week5/results/task_a/task_aRESNET_dim_out_fc_as_image_margin_0.1_lr_0.0001/task_a_triplet_10.pth

# Run task_b example:
# python main.py --task task_b --train True --sweep False --network_image RESNET50 --network_text FastText --batch_size 32 --lr 1e-4 --margin 0.1
# python main.py --task task_b --train True --sweep False --network_image RESNET101 --network_text FastText --batch_size 32 --lr 1e-4 --margin 0.1
# python main.py --task task_b --train True --sweep False --network_image fasterRCNN --network_text FastText --batch_size 32 --lr 1e-4 --margin 0.1

# Run task_c_a example
# Same as task_a but using --network_text BERT




######################### ATENCIÓ!!! TASK_B use batch 32 i 2 epochs!!!!!!! ######################################

if __name__ == '__main__':
    
    # ------------------------------- ARGS ---------------------------------
    parser = argparse.ArgumentParser(description='Week 5 main script')
    parser.add_argument('--task', type=str, default='task_a', help='task_a --> img2txt // task_b --> txt2img') 
    parser.add_argument('--train', type=bool, default=False, help='Train or test')
    parser.add_argument('--sweep', type=bool, default=False, help='Sweep in wandb or not')
    
    # Device
    # parser.add_argument('--gpu', type=int, default=5, help='GPU device id')
    
    # Image
    parser.add_argument('--network_image', type=str, default='fasterRCNN', help='fasterRCNN, RESNET50, RESNET101')
    parser.add_argument('--dim_out_fc', type=int, default=2048, help='Dimension of the output of the fully connected layer (2048 as image, 1000 as txt)')
    
    # Text
    parser.add_argument('--network_text', type=str, default='BERT', help='FastText or BERT')
    parser.add_argument('--weights_text', type=str,
                        default='/ghome/group03/M5-Project/week5/utils/text/fasttext_wiki.en.bin',
                        help='Path to weights of text model')
    
    # Weights of pretrained model
    parser.add_argument('--weights_model', type=str,
                        default='/ghome/group03/M5-Project/week5/results_new/task_a_new/task_a_RESNET50_FastText_dim_out_fc_2048_margin_0.1_lr_0.0001/task_a_triplet_best.pth',
                        help='Path to weights')
    parser.add_argument('--epoch_test', type=str, default='after', help='before or after')
    parser.add_argument('--num_samples_test', type=int, default=1000, help='Number of samples to test (100,500,1000)')
 
    # Training
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')       # TORNAR A POSAR 64
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')  # TORNAR A POSAR 1
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss')

    args = parser.parse_args()
    
    train(args)