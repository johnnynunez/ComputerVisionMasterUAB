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
from torchvision.models import ResNet50_Weights
import matplotlib.pyplot as plt

from dataset.siamese_data import SiameseMITDataset
from models.models import SiameseNet, EmbeddingNet
from utils import metrics, trainer, losses
from utils.early_stopper import EarlyStopper
#import umap
import numpy as np
import matplotlib.pyplot as plt


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

if __name__ == '__main__':
    # ------------------------------- ARGUMENTS --------------------------------
    parser = argparse.ArgumentParser(description='Task B')
    parser.add_argument('--pretrained', type=bool, default=True, help='Use pretrained weights')
    parser.add_argument('--weights', type=str,
                        default='/ghome/group03/M5-Project/week4/Results/Task_b/task_b_siamese.h5',
                        help='Path to weights')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay')
    parser.add_argument('--inference', type=bool, default=True, help='Inference')
    args = parser.parse_args()

    # ------------------------------- PATHS --------------------------------
    env_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(env_path, 'mcv/datasets/MIT_split')

    output_path = os.path.join(env_path, 'M5-Project/week4/Results/Task_b')

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

    train_dataset = ImageFolder(root=os.path.join(dataset_path, 'train'), transform=transforms.Compose([
        ResNet50_Weights.IMAGENET1K_V2.transforms(),
    ]))

    test_dataset = ImageFolder(root=os.path.join(dataset_path, 'test'), transform=transforms.Compose([
        ResNet50_Weights.IMAGENET1K_V2.transforms(),
    ]))

    siamese_train_dataset = SiameseMITDataset(train_dataset, split_name='train')
    siamese_test_dataset = SiameseMITDataset(test_dataset, split_name='test')

    # ------------------------------- DATALOADER --------------------------------
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, **kwargs)

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    siamese_train_loader = DataLoader(siamese_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    siamese_test_loader = DataLoader(siamese_test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # ------------------------------- MODEL --------------------------------
    margin = 1.
   
    embedding_net = EmbeddingNet(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
       
    model = SiameseNet(embedding_net).to(device)

    if args.inference:

        weights = torch.load(args.weights)["state_dict"]

        model.load_state_dict(weights)
        model.eval()
      
    # --------------------------------- TRAINING --------------------------------
    if not args.inference:
        # Train model
        # Loss function
        loss_func = losses.ContrastiveLoss().to(device)  # margin

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

        # Early stoppper
        early_stopper = EarlyStopper(patience=50, min_delta=10)

        # Learning rate scheduler
        # lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)

        log_interval = 5

        trainer.fit(siamese_train_loader, siamese_test_loader, model, loss_func, optimizer, lr_scheduler, args.num_epochs,
                device, log_interval, output_path)

    # Plot emmbeddings
    train_embeddings_cl, train_labels_cl = metrics.extract_embeddings(train_loader, model, device)
    path = os.path.join(output_path, 'train_embeddings.png')
    metrics.plot_embeddings(train_embeddings_cl, train_labels_cl, train_dataset.classes, "Train embeddings", path)

    val_embeddings_cl, val_labels_cl = metrics.extract_embeddings(test_loader, model, device)
    path = os.path.join(output_path, 'val_embeddings.png')
    metrics.plot_embeddings(val_embeddings_cl, val_labels_cl, train_dataset.classes,  "Test embeddings" , path)


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

    metrics.plot_PR_multiclass(train_dataset.classes, val_labels_cl,y_score_knn,"Results/Task_b")

    # get test images from the test dataloader
    test_images = np.zeros((0, 3, 224, 224))
    test_target = np.zeros((0,))
    y_true_test = []
    for i, (data, target) in enumerate(test_loader):
        test_images, test_target = np.concatenate((test_images, data.to("cpu").detach().numpy()), axis=0), np.concatenate((test_target, target.to("cpu").detach().numpy()), axis=0)

    # get train images from the train dataloader
    train_images = np.zeros((0, 3, 224, 224))
    y_true_train = []
    train_target = np.zeros((0,))
    # target dim 1881,
    for i, (data, target) in enumerate(train_loader):
        train_images, train_target = np.concatenate((train_images, data.to("cpu").detach().numpy()), axis=0), np.concatenate((train_labels_cl, target.to("cpu").detach().numpy()), axis=0)


    metrics.plot_retrieval(
        test_images, train_images, test_target, train_target, neighbors, dis, output_dir="Results/Task_b",
        p="CLASS"
    )
    metrics.plot_retrieval(
        test_images, train_images, test_target, train_target, neighbors, dis, output_dir="Results/Task_b",
        p="BEST"
    )
    metrics.plot_retrieval(
        test_images, train_images, test_target, train_target, neighbors, dis, output_dir="Results/Task_b",
        p="WORST"
    )

    """


    # TSNE
    metrics.tsne_features(train_embeddings_cl, train_labels_cl, labels=test_dataset.classes, title = "TSNE Train", output_path="Results/Task_b/tsne_train_embeddings.png")
    metrics.tsne_features(val_embeddings_cl, val_labels_cl, labels=test_dataset.classes, title= "TSNE test", output_path="Results/Task_b/tsne_test_embeddings.png")
            
    #umap
    
    reducer = umap.UMAP(random_state=42)
    reducer.fit(train_embeddings_cl)
    umap_train_embeddings = reducer.transform(train_embeddings_cl)
    umap_val_embeddings = reducer.transform(val_embeddings_cl)

    metrics.plot_embeddings(umap_train_embeddings, train_labels_cl,train_dataset.classes, "UMAP Train", "Results/Task_b/umap_train_embeddings.png")
    metrics.plot_embeddings(umap_val_embeddings, val_labels_cl, train_dataset.classes, "UMAP test", "Results/Task_b/umap_val_embeddings.png")
"""