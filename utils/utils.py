import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm


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


def output_clean_train(DATA_DIR, traingen, outlier_indices):
    # Create output directory
    clean_dir = DATA_DIR + "/clean_train"
    os.makedirs(clean_dir, exist_ok=True)

    # Map class names
    class_indices = traingen.class_indices
    inv_class_indices = {v: k for k, v in class_indices.items()}

    # Flatten list of outlier indices
    outlier_indices_set = set(outlier_indices)

    # Save only non-outlier images
    current_idx = 0
    saved_count = 0

    for i in tqdm(range(len(traingen))):
        batch_imgs, _ = next(traingen)
        batch_filenames = traingen.filenames[
            i * traingen.batch_size : (i + 1) * traingen.batch_size
        ]

        for j, (img, fname) in enumerate(zip(batch_imgs, batch_filenames)):
            if current_idx not in outlier_indices_set:
                # Convert image to uint8
                img_uint8 = (img[:, :, 0] * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_uint8, mode="L")

                # Create class subdirectory
                class_name = fname.split("/")[0]
                class_dir = os.path.join(clean_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # Save image
                save_path = os.path.join(class_dir, os.path.basename(fname))
                pil_img.save(save_path)
                saved_count += 1

            current_idx += 1

    print(f"Saved {saved_count} non-outlier images to {clean_dir}")
