import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights, resnet50
from tqdm import tqdm

from dataset.mit import MITDataset
from utils.checkpoint import save_checkpoint
from utils.early_stopper import EarlyStopper
from utils.metrics import accuracy

dataset_path = '../../mcv/datasets/MIT_split'
# dataset_path = '../../dataset/MIT_split'
num_classes = 8
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.001
EPOCHS = 250

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

# Load the model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# Replace the last fully-connected layer with a new one that outputs 8 classes
model.fc = torch.nn.Linear(model.fc.in_features, 8)

for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False
    print(name, param.requires_grad)

model = model.to(device)

# Load the data
transform_train = transforms.Compose(
    [
        ResNet50_Weights.IMAGENET1K_V2.transforms(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, shear=0, translate=(0, 0.1)),
    ]
)

transform_val = ResNet50_Weights.IMAGENET1K_V2.transforms()

train_dataset = MITDataset(data_dir=dataset_path, split_name='train', transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = MITDataset(data_dir=dataset_path, split_name='test', transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Define the loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Early stoppper
early_stopper = EarlyStopper(patience=50, min_delta=10)

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

train_acc_list = []
train_loss_list = []
val_acc_list = []
val_loss_list = []

# best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')
best_val_acc = 0
total_time = 0
for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    model.train()
    loop = tqdm(train_loader)
    for idx, (images, labels) in enumerate(loop):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        if device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
        else:
            outputs = model(images)
            loss = loss_fn(outputs, labels)

        # ponemos a cero los gradientes
        optimizer.zero_grad()
        # Backprop (calculamos todos los gradientes automáticamente)
        loss.backward()
        # update de los pesos
        optimizer.step()

        train_acc = accuracy(outputs, labels)
        loop.set_description(f"Train: Epoch [{epoch}/{EPOCHS}]")
        loop.set_postfix(loss=loss.item(), train_acc=train_acc)

    print({"epoch": epoch, "train_loss": loss.item()})
    print({"epoch": epoch, "train_accuracy": train_acc})

    train_loss_list.append(loss.item())
    train_acc_list.append(train_acc)

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_acc = 0
        loop = tqdm(val_loader)
        for idx, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            val_loss += loss_fn(outputs, labels)
            val_acc += accuracy(outputs, labels)
            loop.set_description(f"Validation: Epoch [{epoch}/{EPOCHS}]")
            loop.set_postfix(val_loss=val_loss.item(), val_acc=val_acc)

        val_loss = val_loss / (idx + 1)
        val_acc = val_acc / (idx + 1)
        print({"epoch": epoch, "val_loss": val_loss})
        print({"epoch": epoch, "val_accuracy": val_acc})

        val_loss_list.append(float(val_loss))
        val_acc_list.append(float(val_acc))

        #  # Learning rate scheduler
        lr_scheduler.step(val_loss)

    # Early stopping
    if early_stopper.early_stop(val_loss):
        print("Early stopping at epoch: ", epoch)
        break

    # Save the best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        is_best_loss = True
    else:
        is_best_loss = False
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        is_best_acc = True
    else:
        is_best_acc = False

    if is_best_loss or is_best_acc:
        print(
            "Best model saved at epoch: ",
            epoch,
            " with val_loss: ",
            best_val_loss.item(),
            " and val_acc: ",
            best_val_acc,
        )
        save_checkpoint(
            {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_acc': best_val_acc,
                'optimizer': optimizer.state_dict(),
            },
            is_best_loss,
            is_best_acc,
            filename="task_a_finetunning" + '.h5',
        )

    t1 = time.time()
    total_time += t1 - t0
    print("Epoch time: ", t1 - t0)
    print("Total time: ", total_time)

torch.save(model.state_dict(), "Results/Task_a/Task_a_Resnet50_finetuned.pth")

plot_step = 5

plt.figure(figsize=(10, 12), dpi=150)
plt.title("Loss during training", size=18)
plt.plot(
    np.arange(0, EPOCHS, plot_step), train_loss_list[0::plot_step], color="blue", linewidth=2.5, label="Train subset"
)
plt.plot(
    np.arange(0, EPOCHS, plot_step), val_loss_list[0::plot_step], color="orange", linewidth=2.5, label="Val subset"
)
plt.xticks(np.arange(0, EPOCHS, plot_step).astype(int))
plt.xlabel("Epoch", size=12)
plt.ylabel("Loss", size=12)
plt.legend()
plt.savefig("Results/Task_a/plot_loss.png")
plt.close()

plt.figure(figsize=(10, 12), dpi=150)
plt.title("Accuracy during training", size=18)
plt.plot(
    np.arange(0, EPOCHS, plot_step), train_acc_list[0::plot_step], color="blue", linewidth=2.5, label="Train subset"
)
plt.plot(np.arange(0, EPOCHS, plot_step), val_acc_list[0::plot_step], color="orange", linewidth=2.5, label="Val subset")
plt.xticks(np.arange(0, EPOCHS, plot_step).astype(int))
plt.xlabel("Epoch", size=12)
plt.ylabel("Accuracy", size=12)
plt.legend()
plt.savefig("Results/Task_a/plot_accuracy.png")
plt.close()
