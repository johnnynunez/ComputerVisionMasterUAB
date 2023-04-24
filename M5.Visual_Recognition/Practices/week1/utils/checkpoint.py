import torch


def save_checkpoint(state, is_best_loss, is_best_acc, filename):
    if is_best_loss:
        print("Saving best loss model...")
        torch.save(state, 'checkpoints/best_loss_' + filename)
    if is_best_acc:
        print("Saving best accuracy model...")
        torch.save(state, 'checkpoints/best_acc_' + filename)
