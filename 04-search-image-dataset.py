import os

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import clip
from IPython.display import Image
from IPython.core.display import HTML, display

from common import common_path

# Set the paths
features_path = os.path.join(common_path.project_dir, 'unsplash-dataset/lite/features')

# Read the photos table
photos = pd.read_csv(os.path.join(common_path.project_dir,
                                  'data/unsplash-research-dataset-lite-latest/photos.tsv000'), sep='\t', header=0)

# Load the features and the corresponding IDs
photo_features = np.load(os.path.join(features_path, "features.npy"))
photo_ids = pd.read_csv(os.path.join(features_path, "photo_ids.csv"))
photo_ids = list(photo_ids['photo_id'])

# Load the open CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

search_query = "Two dogs playing in the snow"

with torch.no_grad():
    # Encode and normalize the description using CLIP
    text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
    text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

# Retrieve the description vector and the photo vectors
text_features = text_encoded.cpu().numpy()

# Compute the similarity between the descrption and each photo using the Cosine similarity
similarities = list((text_features @ photo_features.T).squeeze(0))

# Sort the photos by their similarity score
best_photos = sorted(zip(similarities, range(photo_features.shape[0])), key=lambda x: x[0], reverse=True)

# Iterate over the top 3 results
for i in range(3):
    # Retrieve the photo ID
    idx = best_photos[i][1]
    photo_id = photo_ids[idx]

    # Get all metadata for this photo
    photo_data = photos[photos["photo_id"] == photo_id].iloc[0]

    # Display the photo
    url = photo_data["photo_image_url"] + "?w=640"
    print('url: %s' % url)
    display(Image(url=url))

    # Display the attribution text
    display(HTML(f'Photo by <a href="https://unsplash.com/@{photo_data["photographer_username"]}?utm_source=NaturalLanguageImageSearch&utm_medium=referral">{photo_data["photographer_first_name"]} {photo_data["photographer_last_name"]}</a> on <a href="https://unsplash.com/?utm_source=NaturalLanguageImageSearch&utm_medium=referral">Unsplash</a>'))
    print()