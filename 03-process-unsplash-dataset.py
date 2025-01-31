import os
import math

from pathlib import Path
import clip
import torch
from PIL import Image
import numpy as np
import pandas as pd

from common import common_path

# Set the path to the photos
# dataset_version = "lite"  # Use "lite" or "full"
# photos_path = Path("unsplash-dataset") / dataset_version / "photos"

photos_path = os.path.join(common_path.project_dir, 'unsplash-dataset/lite/photos')

# List all JPGs in the folder
photos_files = list(Path(photos_path).glob("*.jpg"))

# Print some statistics
print(f"Photos found: {len(photos_files)}")

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# Function that computes the feature vectors for a batch of images
def compute_clip_features(photos_batch):
    # Load all the photos from the files
    photos = [Image.open(photo_file) for photo_file in photos_batch]

    # Preprocess all photos
    photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)

    with torch.no_grad():
        # Encode the photos batch to compute the feature vectors and normalize them
        photos_features = model.encode_image(photos_preprocessed)
        photos_features /= photos_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return photos_features.cpu().numpy()


# Define the batch size so that it fits on your GPU. You can also do the processing on the CPU, but it will be slower.
batch_size = 16

# Path where the feature vectors will be stored
features_path = os.path.join(common_path.project_dir, 'unsplash-dataset/lite/features')

# Compute how many batches are needed
batches = math.ceil(len(photos_files) / batch_size)

# Process each batch
for i in range(batches):
    print(f"Processing batch {i + 1}/{batches}")

    batch_ids_path = os.path.join(features_path, f"{i:010d}.csv")
    batch_features_path = os.path.join(features_path, f"{i:010d}.npy")

    # Only do the processing if the batch wasn't processed yet
    if not os.path.exists(batch_features_path):
        try:
            # Select the photos for the current batch
            batch_files = photos_files[i * batch_size: (i + 1) * batch_size]

            # Compute the features and save to a numpy file
            batch_features = compute_clip_features(batch_files)
            np.save(batch_features_path, batch_features)

            # Save the photo IDs to a CSV file
            photo_ids = [photo_file.name.split(".")[0] for photo_file in batch_files]
            photo_ids_data = pd.DataFrame(photo_ids, columns=['photo_id'])
            photo_ids_data.to_csv(batch_ids_path, index=False)
        except:
            # Catch problems with the processing to make the process more robust
            print(f'Problem with batch {i}')

# Load all numpy files
features_list = [np.load(features_file) for features_file in sorted(Path(features_path).glob("*.npy"))]

# Concatenate the features and store in a merged file
features = np.concatenate(features_list)
np.save(os.path.join(features_path, "features.npy"), features)

# Load all the photo IDs
photo_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(Path(features_path).glob("*.csv"))])
photo_ids.to_csv(os.path.join(features_path, "photo_ids.csv"), index=False)

