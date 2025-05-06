import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from pathlib import Path
import shutil


def visualize_outliers(num_examples, traingen, batch_size, outlier_indices):
    fig, axs = plt.subplots(2, num_examples, figsize=(15, 6))

    # Find indices that are no outliers
    inlier_indices = np.setdiff1d(
        np.arange(len(traingen) * batch_size), outlier_indices
    )
    inlier_indices = inlier_indices[:num_examples]

    counter = 0
    for idx in outlier_indices[:num_examples]:
        # Get the batch index and image index in the batch
        batch_idx = idx // batch_size
        image_idx_in_batch = idx % batch_size

        # Fetch the image from the generator's batch
        batch = traingen[batch_idx]
        original_image = batch[0][image_idx_in_batch]

        # Plot original image
        axs[0, counter].imshow(original_image[:, :, 0], cmap="gray")
        axs[0, counter].set_title("Outlier")
        axs[0, counter].axis("off")

        # Plot non-outlier (inlier) image
        inlier_idx = inlier_indices[counter]
        inlier_batch_idx = inlier_idx // batch_size
        inlier_image_idx = inlier_idx % batch_size
        inlier_image = traingen[inlier_batch_idx][0][inlier_image_idx]

        axs[1, counter].imshow(inlier_image[:, :, 0], cmap="gray")
        axs[1, counter].set_title("Non-Outlier")
        axs[1, counter].axis("off")

        counter += 1

    fig.suptitle("Comparison: Outlier Images vs. Kept Images", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def output_clean_train(DATA_DIR, traingen, outlier_indices):
    src_dir = Path(DATA_DIR) / "train"
    dst_dir = Path(DATA_DIR) / "clean_train"
    os.makedirs(dst_dir, exist_ok=True)

    # Get filenames
    filenames = traingen.filenames
    outlier_set = set(outlier_indices)

    # Copy non-outlier images
    for idx, rel_path in tqdm(enumerate(filenames), total=len(filenames)):
        # If outlier, skip
        if idx in outlier_set:
            continue

        src_path = src_dir / rel_path
        dst_class_dir = dst_dir / rel_path.split("/")[0]
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_class_dir / Path(rel_path).name
        # move file
        shutil.copy2(src_path, dst_path)

    print(f"Copied {len(filenames) - len(outlier_set)} non-outlier images to {dst_dir}")


# plot accuracy
def plot_accuracy(history, plot_title="Model"):
    """
    Helper function to plot accuracy of a model.
    """

    plt.figure(figsize=(12, 6))

    best_val_acc = np.max(history.history["val_accuracy"])
    best_epoch = np.argmax(history.history["val_accuracy"])

    epochs = range(1, len(history.history["accuracy"]) + 1)

    plt.plot(epochs, history.history["accuracy"], label="Training Accuracy")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation Accuracy")

    plt.axvline(
        best_epoch + 1,
        color="red",
        linestyle="--",
        label=f"Best val acc: {best_val_acc:.4f}",
    )

    plt.title(f"{plot_title}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.xticks(epochs)
    plt.legend(loc="lower right")


def plot_accuracy_and_loss(history, plot_title="Model Training History"):
    plt.figure(figsize=(14, 6))

    # Extract metrics from history
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(acc) + 1)

    # Accuracy plot
    plt.subplot(1, 2, 1)
    best_epoch_acc = np.argmax(val_acc) + 1
    best_val_acc = val_acc[best_epoch_acc - 1]

    plt.plot(epochs, acc, label="Training Accuracy", linewidth=2)
    plt.plot(epochs, val_acc, label="Validation Accuracy", linewidth=2)
    plt.axvline(
        x=best_epoch_acc,
        color="k",
        linestyle="--",
        label=f"Best Val Acc: {best_val_acc:.4f} at Epoch {best_epoch_acc}",
    )

    plt.title("Training and Validation Accuracy", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)

    # Loss plot
    plt.subplot(1, 2, 2)
    best_epoch_loss = np.argmin(val_loss) + 1
    best_val_loss = val_loss[best_epoch_loss - 1]

    plt.plot(epochs, loss, label="Training Loss", linewidth=2)
    plt.plot(epochs, val_loss, label="Validation Loss", linewidth=2)
    plt.axvline(
        x=best_epoch_loss,
        color="k",
        linestyle="--",
        label=f"Best Val Loss: {best_val_loss:.4f} at Epoch {best_epoch_loss}",
    )

    plt.title("Training and Validation Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10)

    plt.suptitle(plot_title, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
