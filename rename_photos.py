import os

# Path to your dataset folder
folder_path = "D:/data/thulo_lwang"  # change this to your folder

# Get all files in the folder
files = os.listdir(folder_path)

# Sort files (optional, ensures consistent order)
files.sort()

# Loop through and rename each file
for i, filename in enumerate(files, start=1):
    # Get file extension (e.g., .jpg, .png)
    ext = os.path.splitext(filename)[1]
    # New file name (e.g., 1.jpg, 2.jpg, ...)
    new_name = f"{i}{ext}"
    # Rename file
    os.rename(os.path.join(folder_path, filename),
              os.path.join(folder_path, new_name))

print(" All images renamed successfully!")
