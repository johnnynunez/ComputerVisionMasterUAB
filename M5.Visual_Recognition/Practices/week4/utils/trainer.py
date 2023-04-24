import numpy as np
import torch
from utils.checkpoint import save_checkpoint_loss

# if torch.cuda.is_available():
#     print("CUDA is available")
#     device = torch.device("cuda")
#     torch.cuda.amp.GradScaler()
# elif torch.backends.mps.is_available():
#     print("MPS is available")
#     device = torch.device("cpu")
# else:
#     print("CPU is available")
#     device = torch.device("cpu")

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, device, log_interval, output_path,
        metrics=[],
        start_epoch=0, wandb = None, name=None):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    for epoch in range(0, start_epoch):
        if scheduler is not None:
           scheduler.step()

    best_val_loss = np.inf
    is_best_val_loss = False

    train_loss_list = []
    val_loss_list = []

    for epoch in range(start_epoch, n_epochs):
        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics, name, wandb, epoch)
        train_loss_list.append(train_loss)

        if scheduler is not None:
            scheduler.step()

        message = 'Epoch: {}/{}.           Train set:       Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                         train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())
            
            
        if name == 'task_e':
            # Save the model checkpoint
            print("Saving model at epoch: ", epoch)
            path=output_path + f"/task_e_triplet_{epoch + 1}.pth"
            torch.save(model.state_dict(), path)
        else:
            val_loss, metrics = test_epoch(val_loader, model, loss_fn, device, metrics)
            val_loss /= len(val_loader)
            val_loss_list.append(val_loss)

            message += '\nEpoch: {}/{}.           Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                            val_loss)
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)

            # Save the best model with the lowest validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best_val_loss = True
            else:
                is_best_val_loss = False

            if is_best_val_loss:
                print("Best model saved at epoch: ", epoch, " with val_loss: ", best_val_loss)
                save_checkpoint_loss(
                    {
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'best_val_loss': best_val_loss,
                        'optimizer': optimizer.state_dict(),
                    },
                    is_best_val_loss,
                    path=output_path + "/task_b_siamese.pth"
                )

    path = output_path + "/loss.png"
    plot_loss(train_loss_list, val_loss_list, path)


def train_epoch(train_loader, model, loss_fn, optimizer, device, log_interval, metrics, name, wandb, epoch):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    count = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if device:
            data = tuple(d.to(device) for d in data)
            if target is not None:
                try:
                    target = target.to(device)
                except:
                    a = 1

        optimizer.zero_grad()

        # if name == 'task_e':
        #     target1 = []
        #     target2 = []
        #     target3 = []
        #     for i in range(data[0].shape[0]):
        #         d1 = {}
        #         d1['boxes'] = target[0][i].to(device)
        #         d1['labels'] = target[1][i].to(device)
        #         target1.append(d1)
        #         d2 = {}
        #         d2['boxes'] = target[2][i].to(device)
        #         d2['labels'] = target[3][i].to(device)
        #         target2.append(d2)
        #         d3 = {}
        #         d3['boxes'] = target[4][i].to(device)
        #         d3['labels'] = target[5][i].to(device)
        #         target3.append(d3)
        #     target_in = (target1, target2, target3)
        #     outputs = model(*data, *target_in)
        # else:
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        
        if torch.isnan(loss).any():
            print("NAN loss")
        else:
            total_loss += loss.item()
            losses.append(loss.item())
            count += 1
        
        
        wandb.log({'epoch': epoch, 'batch': batch_idx, 'loss': loss.item()})
        
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = 'Train: [{:>4}/{} ({:2.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())

            print(message)
            losses = []
            

    # total_loss /= (batch_idx + 1)
    total_loss /= count
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, device, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()

        model.eval()
        val_loss = 0

        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if device:
                data = tuple(d.to(device) for d in data)
                if target is not None:
                    target = target.to(device)

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics


def plot_loss(train_loss_list, val_loss_list, output_path):
    import matplotlib.pyplot as plt
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    # x label is epock
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(output_path)
    plt.close()
