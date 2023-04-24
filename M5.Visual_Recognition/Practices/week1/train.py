import time

import torch
import torchvision.transforms as transforms
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from dataset.mit import MITDataset
from models.resnet import ResNet
from utils.checkpoint import save_checkpoint
from utils.early_stopper import EarlyStopper
from utils.metrics import accuracy


def train(args):
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
    # Initialize wandb
    wandb.init(mode=args.wandb)
    # Print wandb.config
    print(wandb.config)

    args.experiment_name = wandb.config.experiment_name

    # Load the model
    model = ResNet()

    # Write model summary to console and WandB
    wandb.config.num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters:", wandb.config.num_params)
    summary(model, input_size=(32, 3, wandb.config.IMG_HEIGHT, wandb.config.IMG_WIDTH))

    # AFTER SUMMARY ALWAYS
    model = model.to(device)
    # if pytorch versions is >=1.13.1 use this line
    if torch.__version__ >= '1.13.1' and device.type == 'cuda':
        model = torch.compile(model)  # Pytorch 2.0

    # Load the data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((wandb.config.IMG_HEIGHT, wandb.config.IMG_WIDTH), antialias=False)]
    )

    # Data augmentation
    if wandb.config.data_augmentation is True:
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, shear=0, translate=(0, 0.1)),
                transforms.ToTensor(),
                transforms.Resize((wandb.config.IMG_HEIGHT, wandb.config.IMG_WIDTH), antialias=False),
            ]
        )

    train_dataset = MITDataset(data_dir=args.dataset_path, split_name='train', transform=transform)
    train_loader = DataLoader(
        train_dataset, batch_size=wandb.config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    val_dataset = MITDataset(data_dir=args.dataset_path, split_name='test', transform=transform)
    val_loader = DataLoader(
        val_dataset, batch_size=wandb.config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=wandb.config.LEARNING_RATE, weight_decay=wandb.config.WEIGHT_DECAY
    )

    # Early stoppper
    early_stopper = EarlyStopper(patience=50, min_delta=10)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_val_acc = 0
    total_time = 0
    for epoch in range(1, wandb.config.EPOCHS + 1):
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
            loop.set_description(f"Train: Epoch [{epoch}/{wandb.config.EPOCHS}]")
            loop.set_postfix(loss=loss.item(), train_acc=train_acc)

        wandb.log({"epoch": epoch, "train_loss": loss.item()})
        wandb.log({"epoch": epoch, "train_accuracy": train_acc})
        wandb.log({"epoch": epoch, "learning_rate": wandb.config.LEARNING_RATE})

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
                loop.set_description(f"Validation: Epoch [{epoch}/{wandb.config.EPOCHS}]")
                loop.set_postfix(val_loss=val_loss.item(), val_acc=val_acc)

            val_loss = val_loss / (idx + 1)
            val_acc = val_acc / (idx + 1)
            wandb.log({"epoch": epoch, "val_loss": val_loss})
            wandb.log({"epoch": epoch, "val_accuracy": val_acc})

            #  # Learning rate scheduler
            lr_scheduler.step(val_loss)
            # log learning rate from scheduler
            wandb.log({"epoch": epoch, "learning_rate": lr_scheduler.optimizer.param_groups[0]['lr']})

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
                filename=wandb.config.experiment_name + '.h5',
            )

        t1 = time.time()
        total_time += t1 - t0
        print("Epoch time: ", t1 - t0)
        print("Total time: ", total_time)

    # model.load_state_dict(best_model_wts)
