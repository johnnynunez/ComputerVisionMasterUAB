import argparse
import os
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from dataset.triplet_data import TripletMITDataset
from models.models import TripletNet, EmbeddingNet
from utils import losses
from utils import metrics
from utils import trainer
from utils.early_stopper import EarlyStopper
from sklearn.metrics import precision_recall_curve
import umap

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task C')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str,
                        default="/ghome/group03/M5-Project/week4/Results/Task_c/task_c_triplet.h5",
                        help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--inference', type=bool, default=True, help='Inference')
    args = parser.parse_args()

    # ------------------------------- PATHS --------------------------------
    env_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(env_path, 'mcv/datasets/MIT_split')

    output_path = os.path.join(env_path, 'M5-Project/week4/Results/Task_c')

    # Create output path if it does not exist
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # -------------------------------- DEVICE --------------------------------
    if torch.cuda.is_available():
        print("CUDA is available")
        device = torch.device("cuda")
        torch.cuda.amp.GradScaler()
    elif torch.backends.mps.is_available():
        print("MPS is available")
        device = torch.device("mps")
    else:
        print("CPU is available")
        device = torch.device("cpu")

    # ------------------------------- DATASET --------------------------------
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    resnet50 = ResNet50_Weights.IMAGENET1K_V2
    resnet18 = ResNet18_Weights.IMAGENET1K_V1

    train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transforms.Compose([
        resnet50.transforms(),
    ]))

    test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transforms.Compose([
        resnet50.transforms(),
    ]))

    triplet_train_dataset = TripletMITDataset(train_dataset, split_name='train')
    triplet_test_dataset = TripletMITDataset(test_dataset, split_name='test')

    # ------------------------------- DATALOADER --------------------------------
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, **kwargs)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    triplet_test_loader = DataLoader(triplet_test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # ------------------------------- MODEL --------------------------------
    margin = args.margin

    embedding_net = EmbeddingNet(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
       
    model = TripletNet(embedding_net).to(device)

    if args.inference:

        weights = torch.load(args.weights)["state_dict"]

        model.load_state_dict(weights)
        model.eval()

    # --------------------------------- TRAINING --------------------------------
    if not args.inference:
        # Loss function
        loss_func = losses.TripletLoss(margin).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Early stoppper
        early_stopper = EarlyStopper(patience=50, min_delta=10)

        # Learning rate scheduler
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)

        log_interval = 5

        trainer.fit(triplet_train_loader, triplet_test_loader, model, loss_func, optimizer, lr_scheduler, args.num_epochs,
                    device, log_interval, output_path)

    # Plot emmbedings
    train_embeddings_cl, train_labels_cl = metrics.extract_embeddings(train_loader, model, device)
    path = os.path.join(output_path, 'train_embeddings.png')
    metrics.plot_embeddings(train_embeddings_cl, train_labels_cl, train_dataset.classes, "Train",  path)
    val_embeddings_cl, val_labels_cl = metrics.extract_embeddings(test_loader, model, device)
    path = os.path.join(output_path, 'val_embeddings.png')
    metrics.plot_embeddings(val_embeddings_cl, val_labels_cl, train_dataset.classes, "Test", path)


    # ------------------------------- METRICS  and retrieval --------------------------------
    # extract number of classes
    num_classes = len(train_dataset.classes)
    
    # Flatten the embeddings to 1D array
    train_embeddings_cl = train_embeddings_cl.reshape(train_embeddings_cl.shape[0], -1)
    val_embeddings_cl = val_embeddings_cl.reshape(val_embeddings_cl.shape[0], -1)

    # Create a KNN classifier
    knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors as needed

    # Fit the KNN classifier to the train embeddings and labels
    knn.fit(train_embeddings_cl, list(range(train_embeddings_cl.shape[0])))

    neighbors = knn.kneighbors(val_embeddings_cl)[1]

    def getMostSimilar(queryFeatures, k=None):

        # Inference every query
        (dis, neighbors) = knn.kneighbors(queryFeatures, return_distance=True)

        return dis, neighbors

    (dis, neighbors) = getMostSimilar(val_embeddings_cl, train_embeddings_cl.shape[0])
    results = []
    for i, label in enumerate(val_labels_cl):
        results.append((train_labels_cl[neighbors[i]] == label))
    results = np.array(results)
    print("P@1: ", metrics.mPrecisionK(results, 1))
    print("P@5: ", metrics.mPrecisionK(results, 5))
    print("MAP: ", metrics.MAP(results))

    clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5, metric = "manhattan"))
    clf.fit(train_embeddings_cl, train_labels_cl)

    y_score_knn = clf.predict_proba(val_embeddings_cl)

    metrics.plot_PR_multiclass(train_dataset.classes, val_labels_cl,y_score_knn,"Results/Task_c")


    #get test images from the test dataloader
    test_images = np.zeros((0, 3, 224, 224))
    y_true_test = []
    for i, (data, target) in enumerate(test_loader):
        test_images = np.concatenate((test_images, data.to("cpu").detach().numpy()), axis=0)

    #get train images from the train dataloader
    train_images = np.zeros((0, 3, 224, 224))
    y_true_train = []
    for i, (data, target) in enumerate(train_loader):
        train_images = np.concatenate((train_images, data.to("cpu").detach().numpy()), axis=0)

    

    metrics.plot_retrieval(
    test_images,train_images, val_labels_cl, train_labels_cl, neighbors, dis, output_dir="Results/Task_c", p="CLASS"
    )
    metrics.plot_retrieval(
        test_images, train_images, val_labels_cl, train_labels_cl, neighbors, dis, output_dir="Results/Task_c", p="BEST"
    )
    metrics.plot_retrieval(
        test_images, train_images, val_labels_cl, train_labels_cl, neighbors, dis, output_dir="Results/Task_c", p="WORST"
    )

    #tsne

    metrics.tsne_features(train_embeddings_cl, train_labels_cl, labels=test_dataset.classes, title = "TSNE Train", output_path="Results/Task_c/tsne_train_embeddings.png")
    metrics.tsne_features(val_embeddings_cl, val_labels_cl, labels=test_dataset.classes, title= "TSNE test", output_path="Results/Task_c/tsne_test_embeddings.png")
            
    #umap
    
    reducer = umap.UMAP(random_state=42)
    reducer.fit(train_embeddings_cl)
    umap_train_embeddings = reducer.transform(train_embeddings_cl)
    umap_val_embeddings = reducer.transform(val_embeddings_cl)

    metrics.plot_embeddings(umap_train_embeddings, train_labels_cl,train_dataset.classes, "UMAP Train", "Results/Task_c/umap_train_embeddings.png")
    metrics.plot_embeddings(umap_val_embeddings, val_labels_cl, train_dataset.classes, "UMAP test", "Results/Task_c/umap_val_embeddings.png")
