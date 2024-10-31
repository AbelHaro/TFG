import zipfile
import os
from datetime import datetime

# Define the paths
TRAIN_NAME =
source_dir = '/content/runs/' + TRAIN_NAME
zip_file_path = '/content/runs/train.zip'

# Get the current date and time
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
new_zip_file_path = f'/content/runs/train_{current_time}.zip'

# Create a zip file with the new name
with zipfile.ZipFile(new_zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            zipf.write(os.path.join(root, file),
                        os.path.relpath(os.path.join(root, file),
                                        os.path.join(source_dir, '..')))

print(f'Zipped files to {new_zip_file_path}')
