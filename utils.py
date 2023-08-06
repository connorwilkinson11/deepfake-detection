def log_training(epoch, stats):
    """Print the train, validation, test accuracy/loss/auroc.

    Each epoch in `stats` should have order
        [val_acc, val_loss, val_auc, train_acc, ...]
    Test accuracy is optional and will only be logged if stats is length 9.
    """
    splits = ["Train", "Test"]
    metrics = ["Accuracy", "Loss"]
    print("Epoch {}".format(epoch))
    for j, split in enumerate(splits):
        for i, metric in enumerate(metrics):
            idx = len(metrics) * j + i
            print(f"\t{split} {metric}:{round(stats[-1][idx],4)}")
