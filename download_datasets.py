# IMPORTANT: This script should be run after cloning / forking the repository to download all datasets and place them under the correct directories
# Please make sure to cd to the root directory of the repository before running
# The script is idempotent, meaning that it can be run multiple times without any side effects. If interrupted, just re-run it. 
# The script can take up to 10 minutes to run on a good internet connection, so feel free to grab some coffee :)

import gdown
import os

# URLs
original_real_url = "https://drive.google.com/uc?id=1pYxg91VziqpNxGfYTI6J6l6uKGi5OpdD"
processed_real_url = "https://drive.google.com/uc?id=1KfG6Mjkxnfhw5As3OgWpFMke8iTGGVFP"
orginal_generated_url = "https://drive.google.com/uc?id=18G1YpsjvL0DmRx95ebsPd4_2PgQSZWKl"
processed_generated_url = "https://drive.google.com/uc?id=1ud2dCGArD8L_0QNPGOFLG-6X9-JXlaJE"

# Target paths
original_real_path = "Datasets PreProcessing/Real life data/Original Real Data.dat.zip"
processed_real_path = "Datasets/Processed Real Data.dat.zip"
original_generated_path = "Datasets PreProcessing/Data Generation/Original Generated Data.dat.zip"
processed_generated_path = "Datasets/Processed Generated Data.dat.zip"

# Download datasets
print("Downloading datasets...")
gdown.download(original_real_url, original_real_path, quiet=False)
gdown.download(processed_real_url, processed_real_path, quiet=False)
gdown.download(orginal_generated_url, original_generated_path, quiet=False)
gdown.download(processed_generated_url, processed_generated_path, quiet=False)

# Unzip datasets in place
print("Unzipping datasets...")
gdown.extractall(original_real_path)
gdown.extractall(processed_real_path)
gdown.extractall(original_generated_path)
gdown.extractall(processed_generated_path)

# Remove zip files
print("Cleaning up zip files...")
os.remove(original_real_path)
os.remove(processed_real_path)
os.remove(original_generated_path)
os.remove(processed_generated_path)

# Confirmation
print("All done and ready to go!")
