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
