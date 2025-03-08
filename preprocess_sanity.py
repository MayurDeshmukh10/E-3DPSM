import os

def find_folders_missing_meta(root_dir):
    missing_meta_folders = []

    # Loop over each item in the given train directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        # Check if the item is a folder
        if os.path.isdir(folder_path):
            meta_file_path = os.path.join(folder_path, 'meta.json')
            # Check if meta.json is missing in this folder
            if not os.path.exists(meta_file_path):
                missing_meta_folders.append(folder)
    
    return missing_meta_folders

if __name__ == "__main__":
    train_directory = '/scratch/inf0/user/mdeshmuk/EE3D-preprocessed/EE3D-S/train'
    
    missing_folders = find_folders_missing_meta(train_directory)
    
    if missing_folders:
        print("Folders missing meta.json file:")
        for folder in missing_folders:
            print(folder)
    else:
        print("All folders have meta.json file.")
