import random

# File path
file_path = '../datasets/hillforts/train.txt'

# Read the lines from the file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Shuffle the lines
random.shuffle(lines)

# Write the shuffled lines back to the file
with open(file_path, 'w') as file:
    file.writelines(lines)