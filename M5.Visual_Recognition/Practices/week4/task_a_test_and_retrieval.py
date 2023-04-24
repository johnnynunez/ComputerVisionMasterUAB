import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from dataset.mit import MITDataset
from utils.metrics import accuracy, plot_retrieval, tsne_features, plot_embeddings
import umap

finetuned = True
dataset_path = '../../mcv/datasets/MIT_split'
num_classes = 8
batch_size = 256

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

if finetuned:
    model = resnet50()

    # Replace the last fully-connected layer with a new one that outputs 8 classes
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 8)

    model.load_state_dict(torch.load("Results/Task_a/Task_a_Resnet50_finetuned.pth"))
else:
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    # Replace the last fully-connected layer with a new one that outputs 8 classes
    fc_in_features = model.fc.in_features
    model.fc = torch.nn.Linear(fc_in_features, 8)

model = model.to(device)

transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
# transform to tensor
# transform = transforms.Compose( [transforms.ToTensor()] )

test_dataset = MITDataset(data_dir=dataset_path, split_name='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

train_dataset = MITDataset(data_dir=dataset_path, split_name='train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

loss_fn = torch.nn.CrossEntropyLoss()
softmax1 = torch.nn.Softmax(dim=1)

y_true_test = []
y_pred = []

model.eval()
with torch.no_grad():
    test_loss = 0
    test_acc = 0
    loop = tqdm(test_loader)
    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_loss += loss_fn(outputs, labels)
        test_acc += accuracy(outputs, labels)

        outputs = softmax1(outputs)

        y_true_test.extend(labels.to("cpu").detach().numpy().flatten().tolist())
        y_pred.extend(np.max(outputs.to("cpu").detach().numpy(), axis=1).flatten().tolist())

    test_loss = float(test_loss / (idx + 1))
    test_acc = float(test_acc / (idx + 1))

    print('Test accuracy:', round(test_acc, 4))
    print('Test loss:', round(test_loss, 4))

y_true_test = np.asarray(y_true_test).flatten()
y_pred = np.asarray(y_pred).flatten()

fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=200)
ax.set_title("Precision-Recall curve", size=16)
for class_id in range(0, num_classes):
    PrecisionRecallDisplay.from_predictions(
        np.where(y_true_test == class_id, 1, 0),
        np.where(y_true_test == class_id, y_pred, 1 - y_pred),
        ax=ax,
        name="Class " + str(test_dataset.classes[class_id]),
    )
plt.savefig("Results/Task_a/PrecisionRecallCurve.png")
plt.close()

# Image retrieval:

model_retrieval = torch.nn.Sequential(*(list(model.children())[:-1]))

y_true_test = []
y_true_train = []

image_features_train = np.zeros((0, fc_in_features))

train_images = np.zeros((0, 3, 224, 224))
with torch.no_grad():
    loop = tqdm(train_loader)
    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_retrieval(images).to("cpu").detach().numpy()
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1]))
        image_features_train = np.concatenate((image_features_train, outputs), axis=0)

        y_true_train.extend(labels.to("cpu").detach().numpy().flatten().tolist())

        print(train_images.shape, images.to("cpu").detach().numpy().shape)
        train_images = np.concatenate((train_images, images.to("cpu").detach().numpy()), axis=0)

y_true_train = np.asarray(y_true_train).flatten()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(np.array(image_features_train), y_true_train)

test_images = np.zeros((0, 3, 224, 224))
image_features_test = np.zeros((0, fc_in_features))
with torch.no_grad():
    loop = tqdm(test_loader)
    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model_retrieval(images).to("cpu").detach().numpy()
        outputs = np.reshape(outputs, (outputs.shape[0], outputs.shape[1]))

        y_true_test.extend(labels.to("cpu").detach().numpy().flatten().tolist())

        image_features_test = np.concatenate((image_features_test, outputs), axis=0)

        test_images = np.concatenate((test_images, images.to("cpu").detach().numpy()), axis=0)

y_true_test = np.asarray(y_true_test).flatten()

compute_neighbors = image_features_train.shape[0]
neigh_dist, neigh_ind = knn.kneighbors(image_features_test, n_neighbors=compute_neighbors, return_distance=True)
neigh_labels = y_true_train[neigh_ind]

# print(y_true_test[0:3], neigh_labels[0:3, 0:5], neigh_dist[0:3, 0:5])

y_true_test_repeated = np.repeat(np.expand_dims(y_true_test, axis=1), compute_neighbors, axis=1)

# We compare class of query image (test) with neighbors (database images of train subset)
prec_at_1 = accuracy_score(y_true_test_repeated[:, 0].flatten(), neigh_labels[:, 0].flatten())
prec_at_5 = accuracy_score(y_true_test_repeated[:, 0:5].flatten(), neigh_labels[:, 0:5].flatten())

print("Prec@1:", prec_at_1)
print("Prec@5:", prec_at_5)

for k in [1, 3, 5, 10, 20, 30]:
    prec_at_k = accuracy_score(y_true_test_repeated[:, 0:k].flatten(), neigh_labels[:, 0:k].flatten())
    print("Prec@" + str(k) + ":", prec_at_k)

aps_all = []
aps_5 = []
for i in range(neigh_labels.shape[0]):
    aps_all.append(average_precision_score((neigh_labels[i] == y_true_test[i]).astype(int), -neigh_dist[i]))
    aps_5.append(average_precision_score((neigh_labels[i, 0:5] == y_true_test[i]).astype(int), -neigh_dist[i, 0:5]))
mAP_all = np.mean(aps_all)
mAP_5 = np.mean(aps_5)

print("mAP@all:", mAP_all)
print("mAP@5:", mAP_5)

plot_retrieval(
    test_images, train_images, y_true_test, y_true_train, neigh_ind, neigh_dist, output_dir="Results/Task_a", p="CLASSd"
)
plot_retrieval(
    test_images, train_images, y_true_test, y_true_train, neigh_ind, neigh_dist, output_dir="Results/Task_a", p="BEST"
)
plot_retrieval(
    test_images, train_images, y_true_test, y_true_train, neigh_ind, neigh_dist, output_dir="Results/Task_a", p="WORST"
)


tsne_features(image_features_train, y_true_train, labels=test_dataset.classes, title = "TSNE Train", output_path="Results/Task_a")
tsne_features(image_features_test, y_true_test, labels=test_dataset.classes, title = "TSNE Test",output_path="Results/Task_a")


reducer = umap.UMAP(random_state=42)
reducer.fit(image_features_train)
umap_train_embeddings = reducer.transform(image_features_train)
umap_val_embeddings = reducer.transform(image_features_test)

plot_embeddings(umap_train_embeddings, y_true_train, train_dataset.classes, "UMAP Train", "Results/Task_a/umap_train_embeddings.png")
plot_embeddings(umap_val_embeddings, y_true_test, train_dataset.classes, "UMAP test", "Results/Task_a/umap_val_embeddings.png")

fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=150)
ax.set_title("Precision-Recall", size=14)
for class_id in range(0, 8):
    PrecisionRecallDisplay.from_predictions(np.where(y_true_test==class_id, 1, 0),
                                            np.where(y_true_train[neigh_ind][:, 0]==class_id, 1, 0),
                                            ax=ax, name="Class " + str([class_id]))
plt.savefig("PR2.png")


test_probas = knn.predict_proba(image_features_test)

fig, ax = plt.subplots(1, 1, figsize=(10,10), dpi=150)
ax.set_title("Precision-Recall", size=14)
for i in range(0, 8):
    PrecisionRecallDisplay.from_predictions(np.where(y_true_test==i, 1, 0),
                                            test_probas[:, i],
                                            ax=ax, name="Class " + str([i]))
plt.savefig("PR3.png")