import torch


def test(model, test_loader, loss_fn):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Transfering images and labels to GPU if available
            labels = labels.to(device)
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Extracting predicted label, and computing validation loss and validation accuracy
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss

    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)

    return v_accuracy, v_loss
