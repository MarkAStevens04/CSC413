import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from Bio.PDB import PDBList, PDBParser
import math
import copy
import esm
import numpy as np
import random
random.seed(777)



def one_hot_encode_column(array, column_index, num_classes):
    """
    Converts a specified column of an array into one-hot encoded vectors.

    82 possible atoms (indiced 0-81) (will use 85 for BOS, EOS, and PAD)
    20 possible aminos (indiced 0-26
    index 82 will be BOS, index 83 will be EOS, index 84 will be PAD

    Parameters:
        array (np.ndarray): Input array.
        column_index (int): The index of the column to one-hot encode.
        num_classes (int): The number of classes for one-hot encoding.

    Returns:
        np.ndarray: A new array where the specified column is replaced by its one-hot encoding.
    """
    # Extract the column to one-hot encode
    indices = array[:, column_index].astype(int)

    # Initialize the one-hot encoded matrix
    one_hot = np.zeros((array.shape[0], num_classes), dtype=int)
    one_hot[np.arange(array.shape[0]), indices] = 1

    # Split the array into parts: before, the one-hot encoding, and after the specified column
    before = array[:, :column_index]
    after = array[:, column_index + 1:]

    # Combine all parts into the final result
    result = np.hstack((before, one_hot, after))
    return result



def normalize_target(target):
    """ Takes a target vector and centers.

    Only takes mean from valid coordinates (final two flags are both true)
    Only updates valid coordinates (leaves 0-fills and -1-fills)

    """
    flags = target[:, -2:]  # Extract the flags
    coords = target[:, -5:-2]  # Extract the coordinates

    # Define a mask for valid coordinates
    valid_mask = (
      (flags[:, -2] != 0) & (flags[:, -1] != 0)
    )

    # Compute the mean using np.mean with the where parameter
    mean_defined = np.mean(coords, axis=0, where=valid_mask[:, np.newaxis])

    # Create a copy of the coordinates for modification
    centered_coords = coords.copy()

    # Subtract the mean only for valid coordinates
    centered_coords[valid_mask, -5:-2] -= mean_defined[-5:-2]

    # Create the centered array
    centered_array = np.hstack((target[:, :-5], centered_coords, target[:, -2:]))

    return centered_array




def format_sample(target):
    """ Formats a single sample (given, target)

    Adds BOS & EOS rows to beginning of sequence and end of sequence.
    Rows are filled entirely with 0s except for the identifier.
    - BOS identifier is index 82
    - EOS identifier is index 83
    - PAD identifier is index 84
    Add as many rows as necessary to make PAD correct sizing.
    """
    # MUST BE MULTIPLE OF 27! Each amino produces 27 atoms
    sequence_length = 540
    # smallest protein is 270, largest is 54102
    # smallest is protein 6VU4

    # index 1 will be BOS, index 25 will be EOS, index 23 will be PAD


    BOS_row = np.zeros(target.shape[1], dtype=int)
    EOS_row = np.zeros(target.shape[1], dtype=int)
    BOS_row[82] = 1
    EOS_row[83] = 1
    if target.shape[0] + 2 < sequence_length:
        PAD = np.zeros((sequence_length - target.shape[0] - 2, target.shape[1]))
        PAD[:, 84] = 1
        # t is our new target
        # t = np.vstack([BOS_row, target, EOS_row, PAD])
        t = np.vstack([target, PAD])
    else:
        # t = np.vstack([BOS_row, target, EOS_row])
        t = np.vstack([target])

    return t





def process_data():
    open_dir = 'PDBs/pre_processed_data'
    save_dir = 'PDBs/processed_data'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'valid'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)

    # Get set of protein codes we have already pre-processed
    code_set = {name[:4] for name in os.listdir(open_dir)}
    code_set = list(code_set)
    # randomize
    random.shuffle(code_set)
    print(f'all names: {code_set}')
    # make our train-test split
    train_codes = code_set[:int(len(code_set) * 0.8)]
    valid_codes = code_set[int(len(code_set) * 0.8):int(len(code_set) * 0.9)]
    test_codes = code_set[int(len(code_set) * 0.9):]

    # Saves processed data into proteins_cleaned under test, train, and valid

    for code in train_codes:
        given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
        target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

        print(f'given: {given}')

        # process data
        onehot_target = one_hot_encode_column(target, 0, 85)
        onehot_target = normalize_target(onehot_target)
        target = format_sample(onehot_target)

        # save input-output pair
        np.save(os.path.join(save_dir, 'train', f'{code}-sample.npy'), given)
        np.save(os.path.join(save_dir, 'train', f'{code}-target.npy'), target)

    for code in valid_codes:
        given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
        target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

        print(f'given: {given}')

        # process data
        onehot_target = one_hot_encode_column(target, 0, 85)
        onehot_target = normalize_target(onehot_target)
        target = format_sample(onehot_target)

        # save input-output pair
        np.save(os.path.join(save_dir, 'valid', f'{code}-sample.npy'), given)
        np.save(os.path.join(save_dir, 'valid', f'{code}-target.npy'), target)

    for code in test_codes:
        given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
        target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

        print(f'given: {given}')

        # process data
        onehot_target = one_hot_encode_column(target, 0, 85)
        onehot_target = normalize_target(onehot_target)
        target = format_sample(onehot_target)

        # save input-output pair
        np.save(os.path.join(save_dir, 'test', f'{code}-sample.npy'), given)
        np.save(os.path.join(save_dir, 'test', f'{code}-target.npy'), target)



def positional_encoding(length, dim, device):
    """
    Encodes information about position of atom.

    Uses sin, cos to create unique combination for each position.
    """
    position = torch.arange(length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float, device=device) *
                         (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe






if __name__ == "__main__":
    process_data()
    code = '5RW2'

    given = np.load(f'PDBs/processed_data/train/{code}-sample.npy', mmap_mode='r', allow_pickle=True)
    target = np.load(f'PDBs/processed_data/train/{code}-target.npy', mmap_mode='r', allow_pickle=True)

    print(f'target shape: {target.shape}')
    print(f'given second: {target[81:150, -5:]}')
    # Fourth column: Whether we know or don't know position
    # Fifth column: Whether we should know the position or not (does the atom exist in the amino or not)

    # output_len = 27 * (90) (there are 27 atoms which we stack into a single row)
    output_len = 2430