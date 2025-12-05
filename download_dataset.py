import gdown
import os

file_id = "1uJ7vJmUNWim5AjWmfC67yW4MT9xqJDwB"
output = "dataset.zip"

# Download file from Google Drive
if not os.path.exists(output):
    gdown.download(f"https://drive.google.com/uc?id={file_id
}", output, quiet=False)
    print("\nContents:", os.listdir("./"))
else:
    print("File already exists.")


#------------------------------------------------------------
#
#Step 2: Unzip in current directory
#------------------------------------------------------------

import zipfile

# path to your dataset zip (update if needed)
zip_path = "dataset.zip"

# extract to current directory
extract_path = "./"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# verify extraction
print("Current Files:", os.listdir(extract_path))
