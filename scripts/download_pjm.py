import kagglehub
import shutil
import os
import glob

# Download latest version
print("Downloading dataset...")
path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")

print("Path to dataset files:", path)

# Define target directory
target_dir = os.path.join(os.getcwd(), "data", "external")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Find PJME file
search_pattern = os.path.join(path, "*PJME*.csv")
files = glob.glob(search_pattern)

if files:
    for f in files:
        filename = os.path.basename(f)
        dest = os.path.join(target_dir, filename)
        shutil.copy2(f, dest)
        print(f"Copied {filename} to {dest}")
else:
    print("PJME file not found in downloaded dataset.")
    # List all files found
    print("Files found:", os.listdir(path))
