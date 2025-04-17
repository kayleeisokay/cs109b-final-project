import numpy as np
import matplotlib.pyplot as plt


def visualize_outliers(num_examples, traingen, batch_size, outlier_indices):
    num_examples = 5
    fig, axs = plt.subplots(2, num_examples, figsize=(15, 6))

    inlier_indices = np.setdiff1d(
        np.arange(len(traingen) * batch_size), outlier_indices
    )
    inlier_indices = inlier_indices[:num_examples]

    counter = 0
    for idx in outlier_indices[:num_examples]:  # Adjust indices if needed
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
