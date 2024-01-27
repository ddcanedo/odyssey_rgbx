import os
import random

def split_data(folder_path, train_ratio=0.8, val_ratio=0.1):
    # Get all file names in the folder
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Shuffle the files
    random.shuffle(files)

    # Calculate the split indices
    train_index = int(len(files) * train_ratio)
    val_index = train_index + int(len(files) * val_ratio)

    # Split the files into training, validation, and testing sets
    train_files = files[:train_index]
    val_files = files[train_index:val_index]
    test_files = files[val_index:]

    # Save the file names to train.txt, val.txt, and test.txt
    with open('../datasets/hillforts/train.txt', 'w') as f:
        for file in train_files:
            file_name = os.path.splitext(file)[0]  # Removes the file extension
            f.write(file_name + '\n')

    with open('../datasets/hillforts/val.txt', 'w') as f:
        for file in val_files:
            file_name = os.path.splitext(file)[0]  # Removes the file extension
            f.write(file_name + '\n')

    with open('../datasets/hillforts/test.txt', 'w') as f:
        for file in test_files:
            file_name = os.path.splitext(file)[0]  # Removes the file extension
            f.write(file_name + '\n')

def semi_manual_split(folder_path, train_ratio=0.9, val_ratio=0.1):
     # Get all file names in the folder
    og_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Shuffle the files
    random.shuffle(og_files)
    files = []
    for file in og_files:
        if file.startswith('PNPG'):
            files.append(file)

    '''with open('datasets/hillforts/test.txt', 'w') as f:
        for file in files:
            file_name = os.path.splitext(file)[0]  # Removes the file extension
            if file_name.startswith('PNPG'):
                f.write(file_name + '\n')

    files = [file for file in files if not file.startswith('PNPG')]'''            

    # Calculate the split indices
    train_index = int(len(files) * train_ratio)
    val_index = train_index + int(len(files) * val_ratio)

    # Split the files into training, validation, and testing sets
    train_files = files[:train_index]
    val_files = files[train_index:]

    # Save the file names to train.txt, val.txt, and test.txt
    with open('../datasets/hillforts/train.txt', 'a') as f:
        for file in train_files:
            file_name = os.path.splitext(file)[0]  # Removes the file extension
            f.write(file_name + '\n')

    with open('../datasets/hillforts/val.txt', 'a') as f:
        for file in val_files:
            file_name = os.path.splitext(file)[0]  # Removes the file extension
            f.write(file_name + '\n')


#split_data('datasets/hillforts/Masks/')
semi_manual_split('../datasets/hillforts/Masks/')