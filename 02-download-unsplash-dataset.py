import os
import urllib.request
from IPython.core.display import HTML, display

from pathlib import Path
import pandas as pd

from common import common_path

# dataset_version = "lite"  # either "lite" or "full"
# unsplash_dataset_path = Path("unsplash-dataset") / dataset_version
unsplash_dataset_path = os.path.join(common_path.project_dir, 'data/unsplash-research-dataset-lite-latest')

# Read the photos table
photos = pd.read_csv(os.path.join(unsplash_dataset_path, "photos.tsv000"), sep='\t', header=0)

# Extract the IDs and the URLs of the photos
photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()

# Print some statistics
print(f'Photos in the dataset: {len(photo_urls)}')

# Path where the photos will be downloaded
# photos_donwload_path = unsplash_dataset_path / "photos"
photos_donwload_path = os.path.join(common_path.project_dir, 'unsplash-dataset/lite/photos')

# Function that downloads a single photo
def download_photo(photo):
    # Get the ID of the photo
    photo_id = photo[0]

    # Get the URL of the photo (setting the width to 640 pixels)
    photo_url = photo[1] + "?w=640"

    # Path where the photo will be stored
    photo_path = os.path.join(photos_donwload_path, (photo_id + ".jpg"))

    # Only download a photo if it doesn't exist
    if not os.path.exists(photo_path):
        try:
            urllib.request.urlretrieve(photo_url, photo_path)
        except:
            # Catch the exception if the download fails for some reason
            print(f"Cannot download {photo_url}")
            pass

from multiprocessing.pool import ThreadPool

# Create the thread pool
threads_count = 16
pool = ThreadPool(threads_count)

# Start the download
pool.map(download_photo, photo_urls)

# Display some statistics
display(f'Photos downloaded: {len(photos)}')
