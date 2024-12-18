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
from torch.utils.data import DataLoader
import torch
import logging
import logging.handlers
import multiprocessing_logging
import time
import parse_seq
import sys
# random.seed(777)

esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_batch_converter = esm_alphabet.get_batch_converter()

block_size = 1000


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def setup(node_name=None):
    # logger.info(f'Setting up...')
    os.makedirs(f'PDBs', exist_ok=True)
    os.makedirs(f'PDBs/all', exist_ok=True)
    os.makedirs(f'PDBs/big_data', exist_ok=True)
    os.makedirs(f'PDBs/pre_processed_data', exist_ok=True)
    os.makedirs(f'PDBs/Residues', exist_ok=True)

    # Download standard aminos if not already downloaded
    if len([name for name in os.listdir('PDBs/Residues')]) == 0:
        import amino_expert
        amino_expert.download_all_aminos()


    if node_name is not None:
        os.makedirs(f'models', exist_ok=True)
        os.makedirs(f'models/{node_name}', exist_ok=True)
        os.makedirs(f'Logs', exist_ok=True)
        os.makedirs(f'Logs/{node_name}', exist_ok=True)
    else:
        os.makedirs(f'models', exist_ok=True)
        os.makedirs(f'Logs', exist_ok=True)

    # Make a trial_tracker to keep our model numbers from conflicting
    try:
        with open(f'trial_tracker.txt', 'x') as file:
            file.write(f'0')
    except FileExistsError:
        # File has already been created
        pass













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





class protein_unifier():
    """
    "Unifies" the proteins and appends into a single file
    """
    def __init__(self, num_looking, name='train'):
        self.in_file = None
        self.out_file = None
        self.save_path = 'PDBs/big_data/'
        self.num_looking = num_looking
        self.track = 0
        self.chunk_size = 10
        self.name = name

    def add_protein(self, sequence, protein):

        if self.out_file is None:
            # self.out_file = np.zeros((protein.shape[0] * self.num_looking, protein.shape[1]))
            # self.out_file = np.save(, allow_pickle=True, arr=self.out_file)
            self.out_file = np.memmap(f'{self.save_path}out-{self.name}', dtype='float32', mode='w+', shape=(protein.shape[0] * self.num_looking, protein.shape[1]))
            self.out_file[:protein.shape[0], :] = protein

            # self.out_file = protein
            # print(f'out shape: {self.out_file.shape}, pshape: {protein.shape}')

        else:
            # self.out_file = np.vstack([self.out_file, protein])
            self.out_file[(self.track) * protein.shape[0]:(self.track + 1) * protein.shape[0], :] = protein

            # print(f'out shape: {self.out_file.shape}, pshape: {protein.shape}')

        # tokenize our sequence before saving!
        seq = ''
        p = 0
        for s in sequence:
            if s == 'b':
                pass
            elif s == 'e':
                pass
            elif s == 'p':
                p += 1
            else:
                seq += s

        seq = seq + '<pad>' * p

        batch = [("protein", seq)]
        _, _, tokens = esm_batch_converter(batch)
        sequence = tokens[0, :].numpy()

        if self.in_file is None:
            # self.in_file = sequence
            # print(f'sequence shape: {sequence.shape}')
            self.in_file = np.memmap(f'{self.save_path}in-{self.name}', dtype='int', mode='w+',
                                      shape=(sequence.shape[0] * self.num_looking,))
            self.in_file[:sequence.shape[0]] = sequence
            self.track += 1
        else:
            # self.in_file = np.hstack([self.in_file, sequence])
            self.in_file[(self.track) * sequence.shape[0]:(self.track + 1) * sequence.shape[0]] = sequence
            self.track += 1


        # If our current step is not the same as our sequence...
        # if (self.track * protein.shape[0]) != len(self.in_file):
        #     print(f'input and output files are out of sync!')
        #     print(f'in_file shape: {len(self.in_file)}')
        #     print(f'out_file shape: {(self.track * protein.shape[0])}')
        #     logging.warning(f'input and output files are out of sync')
        #     logging.info(f'in_file shape: {len(self.in_file)}')
        #     logging.info(f'out_file shape: {self.out_file.shape}')

    def save(self, name):
        # Name should be XXX.bin
        # print(f'in: {self.in_file}')
        # print(f'out: {self.out_file[50, 85:90]}')
        # for t, i in enumerate(self.out_file):
        #     print(f'line {t}: {i[85:90]}')

        # for t, i in enumerate(self.in_file):
        #     print(f'line {t}: {i}')


        np.save(f'{self.save_path}in-{name}', allow_pickle=True, arr=self.in_file)
        np.save(f'{self.save_path}out-{name}', allow_pickle=True, arr=self.out_file)

        # just closes the output file!
        del self.out_file
        del self.in_file

    def __repr__(self):
        msg = 'PROTEIN UNIFIER: \n' + f'total accumulated sequence length: {len(self.in_file)} \n' + f'total accumulated protein  length: {self.out_file.shape} \n'
        return msg



def format_sample(target, pad=False):
    """ Formats a single sample (given, target)

    Adds BOS & EOS rows to beginning of sequence and end of sequence.
    Rows are filled entirely with 0s except for the identifier.
    - BOS identifier is index 82
    - EOS identifier is index 83
    - PAD identifier is index 84
    Add as many rows as necessary to make PAD correct sizing.
    """
    sequence_length = block_size


    BOS_row = np.zeros(target.shape[1], dtype=int)
    EOS_row = np.zeros(target.shape[1], dtype=int)
    BOS_row[82] = 1
    EOS_row[83] = 1

    if target.shape[0] > sequence_length:
        print(f'uh oh! Possibly cutting off values')
        print(f'target length: {target.shape[0]}')
        print(f'max length: {sequence_length}')
        logger.warning(f'uh oh! Possibly cutting off values')
        logger.info(f'target length: {target.shape[0]}')
        logger.info(f'max length: {sequence_length}')


    if target.shape[0] + 2 < sequence_length:
        PAD = np.zeros((sequence_length - target.shape[0] - 2, target.shape[1]))
        PAD[:, 84] = 1
        # t is our new target
        if pad:
            t = np.vstack([BOS_row, target[:sequence_length - 2], PAD, EOS_row])
        else:
            t = np.vstack([BOS_row, target[:sequence_length - 2], PAD, EOS_row])
    else:
        t = np.vstack([BOS_row, target[:sequence_length - 2], EOS_row])

    return t



def stack_atoms(target):
    """
    Turns our 27-dimensional array into a stack!
    Run this BEFORE running format_sample!
    :param target:
    :return:
    """
    # stack groups of 27 atoms into a single row!
    N, M = target.shape
    reshaped_target = target.reshape(N // 27, 27, M)  # Split into groups of 27 rows
    stacked_target = reshaped_target.reshape(N // 27, M * 27)
    # stacked_target = reshaped_target.transpose(0, 2, 1).reshape(N // 27, M * 27)
    # print(f'stacked: {stacked_target.shape}')
    return stacked_target




def format_input(target, pad=False):
    """ Formats a single sample (given, target)

    Adds BOS & EOS rows to beginning of sequence and end of sequence.
    Rows are filled entirely with 0s except for the identifier.
    - BOS identifier is index 82
    - EOS identifier is index 83
    - PAD identifier is index 84
    Add as many rows as necessary to make PAD correct sizing.
    """
    # MUST BE MULTIPLE OF 27! Each amino produces 27 atoms
    sequence_length = block_size
    # smallest protein is 270, largest is 54102
    # smallest is protein 6VU4

    # index 1 will be BOS, index 25 will be EOS, index 23 will be PAD

    BOS = 'b'
    EOS = 'e'
    PAD = 'p'

    target = str(target).upper()
    if len(target) > sequence_length + 2:
        logger.warning(f'UH OH!!! CUTTING OFF VALUES!')
        logger.info(f'len target: {len(target)}')
        logger.info(f'max size: {sequence_length}')

    if len(target) + 2 < sequence_length:
        padding = PAD * (sequence_length - len(target) - 2)
        # t is our new target
        if pad:
            t = BOS + target[:sequence_length - 2] + padding + EOS
        else:
            t = BOS + target[:sequence_length - 2] + EOS
    else:
        t = BOS + target[:sequence_length - 2] + EOS


    return t




def process_data(max_proteins=1000):
    open_dir = 'PDBs/pre_processed_data'
    save_dir = 'PDBs/processed_data'

    os.makedirs(save_dir, exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'train'), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'valid'), exist_ok=True)
    # os.makedirs(os.path.join(save_dir, 'test'), exist_ok=True)

    # Get set of protein codes we have already pre-processed
    code_set = {name[:4] for name in os.listdir(open_dir)}
    code_set = list(code_set)
    # randomize
    random.shuffle(code_set)
    code_set = code_set[:max_proteins]
    print(f'all names: {code_set}')
    # make our train-test split
    train_codes = code_set[:int(len(code_set) * 0.8)]
    valid_codes = code_set[int(len(code_set) * 0.8):int(len(code_set) * 0.9)]
    test_codes = code_set[int(len(code_set) * 0.9):]

    # train_codes = code_set[:]
    # valid_codes = []
    # test_codes = []

    # Saves processed data into proteins_cleaned under test, train, and valid
    pu = protein_unifier(len(train_codes), name='train')
    for i, code in enumerate(train_codes):
        try:
            print(f'Saved one! {code} {round(((i / len(train_codes)) * 100), 2)}')
            given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
            target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

            # print(f'given: {given}')

            # process data
            onehot_target = one_hot_encode_column(target, 0, 85)
            onehot_target = normalize_target(onehot_target)
            # print(f'target shape: {onehot_target.shape}')
            # target = format_sample(onehot_target)
            #
            #
            # # save input-output pair
            # np.save(os.path.join(save_dir, 'train', f'{code}-sample.npy'), given)
            # np.save(os.path.join(save_dir, 'train', f'{code}-target.npy'), target)
            # # print(f'given shape: {len(str(given))}')
            # # print(f'target shape: {onehot_target.shape}')
            g = format_input(given, pad=True)
            t = stack_atoms(onehot_target)
            t = format_sample(t, pad=True)
            # print(f'given after formatting: {len(g)}')
            # print(f'target after formatting: {t.shape}')
            pu.add_protein(g, t)
            # global num_training
            # num_training += 1
        except:
            logger.exception(f'FATAL ERROR WITH THIS PROTEIN! {code}')
    # print(pu)
    pu.save('train')

    pu = protein_unifier(len(valid_codes), name='valid')
    for code in valid_codes:
        given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
        target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

        # print(f'code: {code}')
        # print(f'given: {given}')

        # process data
        onehot_target = one_hot_encode_column(target, 0, 85)
        onehot_target = normalize_target(onehot_target)
        # target = format_sample(onehot_target)
        #
        # # save input-output pair
        # np.save(os.path.join(save_dir, 'valid', f'{code}-sample.npy'), given)
        # np.save(os.path.join(save_dir, 'valid', f'{code}-target.npy'), target)
        g = format_input(given, pad=True)
        t = stack_atoms(onehot_target)
        t = format_sample(t, pad=True)
        # print(f'given after formatting: {len(g)}')
        # print(f'target after formatting: {t.shape}')
        pu.add_protein(g, t)
    # print(pu)
    pu.save('valid')

    pu = protein_unifier(len(test_codes), name='test')
    for code in test_codes:
        given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
        target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

        print(f'given: {given}')

        # process data
        onehot_target = one_hot_encode_column(target, 0, 85)
        onehot_target = normalize_target(onehot_target)
        # target = format_sample(onehot_target)
        #
        # # save input-output pair
        # np.save(os.path.join(save_dir, 'test', f'{code}-sample.npy'), given)
        # np.save(os.path.join(save_dir, 'test', f'{code}-target.npy'), target)
        g = format_input(given, pad=True)
        t = stack_atoms(onehot_target)
        t = format_sample(t, pad=True)
        # print(f'given after formatting: {len(g)}')
        # print(f'target after formatting: {t.shape}')
        pu.add_protein(g, t)
    # print(pu)
    pu.save('test')



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



def custom_collate_fn(batch):
    """
    Collate function to handle variable-length (esm_emb, coords) pairs.
    Each item in batch is (esm_emb, coords) with shape [L, D] and [L, 3], respectively.

    Steps:
    1. Find max_length across all samples in the batch.
    2. Pad all embeddings and coords to this max_length.
    3. Stack them along the batch dimension.
    """
    # ******************* Extend this by adding your own stacking function!!! *******************
    # So first stack all atoms from all amino acids for all proteins.
    # Then, add BOS & EOS identifiers.
    # Then stack all proteins on each other.
    # Then, save this super stack as a single file.
    # Finally, load memory maps from this super stack.

    # batch: list of (esm_emb, coords)
    # esm_emb: [L, D], coords: [L, output_len (27 atoms stacked horizontally)]
    # sets the block size to actually be max_length?
    block_size = 570

    lengths = [item[0].size(0) for item in batch]  # lengths of each sequence
    max_length = max(lengths)

    # Determine embedding dimension (D) from the first sample
    D = batch[0][0].size(-1)

    # Prepare padded tensors
    # We'll pad with zeros for both embeddings and coords
    # After padding:
    # esm_emb_padded: [B, max_length, D]
    # coords_padded: [B, max_length, 3]
    B = len(batch)

    esm_emb_padded = torch.zeros(B, max_length, D)
    coords_padded = torch.zeros(B, max_length, output_len)

    for i, (esm_emb, coords) in enumerate(batch):
        # print(f'esm_emb shape: {esm_emb.shape}')
        # print(f'coords shape: {coords.shape}')
        L = esm_emb.size(0)
        esm_emb_padded[i, :L, :] = esm_emb
        coords_padded[i, :L, :] = torch.from_numpy(coords)
    return esm_emb_padded, coords_padded







def custom_collate_fn_two(batch):
    """
    Collate function to handle variable-length (esm_emb, coords) pairs.
    Each item in batch is (esm_emb, coords) with shape [L, D] and [L, 3], respectively.

    Steps:
    1. Find max_length across all samples in the batch.
    2. Pad all embeddings and coords to this max_length.
    3. Stack them along the batch dimension.
    """
    # ******************* Extend this by adding your own stacking function!!! *******************
    # So first stack all atoms from all amino acids for all proteins.
    # Then, add BOS & EOS identifiers.
    # Then stack all proteins on each other.
    # Then, save this super stack as a single file.
    # Finally, load memory maps from this super stack.


    # print(f'batch[0]: {batch[0]}')
    x = torch.zeros(len(batch), batch[0][0].shape[1], batch[0][0].shape[2])
    t = torch.zeros(len(batch), batch[0][1].shape[0], batch[0][1].shape[1])
    # print(f'x shape: {x.shape}')
    # print(f't shape: {t.shape}')

    for i, (esm_emb, coords) in enumerate(batch):
        # print(f'coords: {coords.shape}')
        # print(f'esm_emb: {esm_emb.shape}')
        x[i, :, :] = esm_emb
        t[i, :, :] = coords

    if device == 'cuda':
        # pin arrays x,t, which allows us to move them to GPU asynchronously
        #  (non_blocking=True)
        x, t = x.pin_memory().to(device, non_blocking=True), t.pin_memory().to(device, non_blocking=True)
    else:
        x, t = x.to(device), t.to(device)
    return x, t











def get_batch(seq, tar, block_size, batch_size, device):
    """
        Return a minibatch of data. This function is not deterministic.
        Calling this function multiple times will result in multiple different
        return values.

        Parameters:
            `data` - a numpy array (e.g., created via a call to np.memmap)
            `block_size` - the length of each sequence
            `batch_size` - the number of sequences in the batch
            `device` - the device to place the returned PyTorch tensor

        Returns: A tuple of PyTorch tensors (x, t), where
            `x` - represents the input tokens, with shape (batch_size, block_size)
            `y` - represents the target output tokens, with shape (batch_size, block_size)
        """

    ix = torch.randint(seq.shape[0] - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((seq[i:i + block_size])) for i in ix])
    t = torch.stack([torch.from_numpy((tar[i:i + block_size, :])) for i in ix])
    # print(f'x batch shape: {x.shape}')
    # print(f't batch shape: {t.shape}')

    if device == 'cuda':
        # pin arrays x,t, which allows us to move them to GPU asynchronously
        #  (non_blocking=True)
        x, t = x.pin_memory().to(device, non_blocking=True), t.pin_memory().to(device, non_blocking=True)
    else:
        x, t = x.to(device), t.to(device)
    return x, t







# Example usage:
# Adjust your dataset so that it no longer pads/truncates sequences.
# Return raw esm_emb and coords at their natural length.
class ProteinStructureDataset(Dataset):
    def __init__(self, pdb_dir, esm_model, esm_batch_converter, train_seq, train_tar, device, num_training):
        self.pdb_dir = pdb_dir
        self.esm_model = esm_model
        self.esm_batch_converter = esm_batch_converter
        self.train_seq = train_seq
        self.train_tar = train_tar
        self.block_size = block_size

        self.device = device
        self.traversed = 0

        self.starts = np.where(self.train_seq == 0)[0]
        self.num_training = num_training

        print(f'num training: {num_training}')

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.num_batches = self.num_training // self.batch_size
        # print(f'self starts: {self.starts}')
        self.update_batches()

    def update_batches(self):
        self.random_seq = torch.randint(self.starts.shape[0], (self.starts.shape[0],))
        # indices = np.linspace(0, self.starts.shape[0])
        indices = np.arange(0, self.starts.shape[0], step=1)
        np.random.shuffle(indices)
        # print(f'indices: {indices}')
        # print(f'batch size: {self.batch_size}')
        self.batches = torch.split(torch.from_numpy(indices), self.batch_size)
        # print(f'self.batches: {self.batches}')

    def __len__(self):
        return self.num_training

    def __iter__(self):
        return self

    # def __next__(self):
    #     if self.traversed >= self.num_batches:
    #         self.update_batches()
    #         self.traversed = 0
    #         raise StopIteration
    #     else:
    #
    #         # print(f'getting item...')
    #         # print(f'code: {self.pdb_files[idx]}')
    #         # pdb_path = f'PDBs/processed_data/train/{self.pdb_files[idx]}'
    #         #
    #         #
    #         # # pdb_path = os.path.join(self.pdb_dir, self.pdb_files[idx])
    #         # seq, coords = self.get_sequence_and_coords(pdb_path)
    #         # seq = str(seq)
    #         #
    #         # # Obtain ESM embeddings for the raw sequence length
    #         # batch = [("protein", seq)]
    #         # _, _, tokens = self.esm_batch_converter(batch)
    #         # tokens = tokens.to(next(self.esm_model.parameters()).device)
    #         # with torch.no_grad():
    #         #     results = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers])
    #         # # Exclude CLS token
    #         # esm_emb = results["representations"][self.esm_model.num_layers][0, 1:len(seq)+1, :]
    #         # # print(f'esm_emb shape: {esm_emb.shape}')
    #         # # print(f'coords shape: {coords.shape}')
    #         #
    #         # # No padding/truncation here. Just return raw.
    #
    #
    #
    #         x, t = self.get_batch(self.train_seq, self.train_tar, self.block_size, self.batch_size, self.device, self.traversed)
    #
    #
    #         # print(f'x prev shape: {x.shape}')
    #
    #         # print(f'x new shape: {x.shape}')
    #         # print(f'train seq: {x.shape}')
    #         # print(f'train tar: {t.shape}')
    #         # print(f'num batches: {self.num_batches}')
    #
    #         self.traversed += 1
    #
    #
    #         return x, t

    def __getitem__(self, idx):

        coords = torch.from_numpy((self.train_tar[idx:idx + block_size]))
        seq = torch.from_numpy((self.train_seq[idx:idx + block_size]))
        seq = seq[None, :]
        # print(f'seq shape pre: {seq.shape}')
        esm_emb = self.to_embedding(seq)
        # print(f'seq shape post: {seq.shape}')


        return esm_emb, coords

    def get_batch(self, seq, tar, block_size, batch_size, device, traversed):
        """
            Return a minibatch of data. This function is not deterministic.
            Calling this function multiple times will result in multiple different
            return values.

            Parameters:
                `data` - a numpy array (e.g., created via a call to np.memmap)
                `block_size` - the length of each sequence
                `batch_size` - the number of sequences in the batch
                `device` - the device to place the returned PyTorch tensor

            Returns: A tuple of PyTorch tensors (x, t), where
                `x` - represents the input tokens, with shape (batch_size, block_size)
                `y` - represents the target output tokens, with shape (batch_size, block_size)
            """
        # print(f'possible start pos: {start_pos}')
        # print(f'seq: {seq}')
        # ix = torch.randint(seq.shape[0] - block_size, (batch_size,))
        # starts = torch.randint(self.starts.shape[0], (self.batch_size,))
        starts = self.batches[traversed]

        # print(f'random seq: {starts}')
        # starts = self.random_seq[(batch_size * traversed):(batch_size * (traversed + 1))]
        # print(f'starts: {starts}')
        # print(f'starts: {self.starts}')
        # print(f'starts: {starts}')
        ix = self.starts[starts]
        # print(f'ix: {ix}')
        # pick from a random start position!
        # print(f'seq size: {seq.shape}')
        x = torch.stack([torch.from_numpy((seq[i:i + block_size])) for i in ix])
        t = torch.stack([torch.from_numpy((tar[i:i + block_size, :])) for i in ix])
        # print(f'x: {x.shape}')
        # print(f't: {t.shape}')
        # print(f'ix: {ix}')
        # print(f'x: {x[1, 100:110]}')
        # for i in range(27):
        #     # print(f'i {i}')
        #     print(f't: {t[1, 107, (90 * i +82):(90 * i + 90)]}')
            # print()
        # print()
        # print()

        # print(f'tar shape: {tar.shape}')
        # print(f't shape: {t[5, 50:55, 87]}')
        # print(f'x batch shape: {x.shape}')
        # print(f't batch shape: {t.shape}')
        # print(f'device: {device}')
        if device == 'cuda':
            # pin arrays x,t, which allows us to move them to GPU asynchronously
            #  (non_blocking=True)
            x, t = x.pin_memory().to(device, non_blocking=True), t.pin_memory().to(device, non_blocking=True)
        else:
            x, t = x.to(device), t.to(device)
        return x, t




    def to_embedding(self, tokens):
        """
        Converts a batch of tokens to a proper embedding!
        # Input: (B, L, 1)
        # Output: (B, L, Embedding_size)

        :param indices:
        :return:
        """
        # print(f'tokens: {tokens[0].shape}')
        tokens = tokens.to(next(self.esm_model.parameters()).device)
        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers])
        esm_emb = results["representations"][self.esm_model.num_layers]
        # print(f'embedded! {esm_emb.shape}')
        # print(f'embedding shape: {esm_emb.shape}')
        return esm_emb



    def get_sequence_and_coords(self, pdb_path):
        # Extract raw sequence and coords
        given = np.load(f'{pdb_path}-sample.npy', mmap_mode='r', allow_pickle=True)
        target= np.load(f'{pdb_path}-target.npy', mmap_mode='r', allow_pickle=True)
        # stack groups of 27 atoms into a single row!
        N, M = target.shape
        # print(f'target shape: {target.shape}')
        # print(target)
        reshaped_target = target.reshape(N // 27, 27, M)  # Split into groups of 27 rows
        stacked_target = reshaped_target.transpose(0, 2, 1).reshape(N // 27, M * 27)
        return given, stacked_target

    def three_letter_to_one(self, res_name):
        mapping = {
            'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H',
            'ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q',
            'ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'
        }
        return mapping.get(res_name, None)












class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, L_q, D = query.size()
        B, L_k, D = key.size()
        B, L_v, D = value.size()

        q = self.query_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1,2)
        k = self.key_proj(key).view(B, L_k, self.num_heads, self.head_dim).transpose(1,2)
        v = self.value_proj(value).view(B, L_v, self.num_heads, self.head_dim).transpose(1,2)

        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)
        context = context.transpose(1,2).contiguous().view(B, L_q, D)
        out = self.out_proj(context)
        return out, attn











class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio*embed_dim), embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln1(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        x = self.ln2(x)
        return x








class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super(CrossAttentionBlock, self).__init__()
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(mlp_ratio*embed_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio*embed_dim), embed_dim),
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, query, context):
        cross_out, _ = self.cross_attn(query, context, context)
        query = query + cross_out
        query = self.ln1(query)
        mlp_out = self.mlp(query)
        query = query + mlp_out
        query = self.ln2(query)
        return query







class ProteinStructurePredictor(nn.Module):
    def __init__(self, embed_dim=768, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1, num_structure_tokens=128):
        super(ProteinStructurePredictor, self).__init__()
        self.embed_dim = embed_dim
        self.num_structure_tokens = num_structure_tokens

        self.structure_tokens = nn.Parameter(torch.randn(1, num_structure_tokens, embed_dim))

        self.encoder_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        self.cross_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)
        ])

        self.residue_decoder = nn.Linear(embed_dim, embed_dim)
        self.coord_predictor = nn.Linear(embed_dim, output_len)

    def forward(self, seq_emb):
        B, L, D = seq_emb.size()
        device = seq_emb.device
        seq_emb = seq_emb + positional_encoding(L, D, device).unsqueeze(0)

        x = seq_emb
        for block in self.encoder_blocks:
            x = block(x)

        structure_tokens = self.structure_tokens.repeat(B, 1, 1)
        structure_tokens = structure_tokens + positional_encoding(self.num_structure_tokens, D, device).unsqueeze(0)

        s = structure_tokens
        for cross_block in self.cross_blocks:
            s = cross_block(s, x)

        s_pooled = s.mean(dim=1, keepdim=True)
        decoded = self.residue_decoder(x + s_pooled)
        coords_pred = self.coord_predictor(decoded)
        return coords_pred




class RMSDLoss(nn.Module):
    def __init__(self):
        super(RMSDLoss, self).__init__()

    def forward(self, pred_coords, true_coords):
        # Create a mask to ignore padded regions
        mask = (true_coords.sum(dim=-1) != 0)  # Assuming padded coords are all zeros
        # print(f'mask shape: {mask.shape}')
        # mask shape: torch.Size([10, 1000])
        # print(f'mask: {mask}')

        # torch.Size([2, 463, 2430])
        # pred_coords has size (batch, block, output)
        # print(f'pred coords: {pred_coords.shape}')
        # print(f'true coords: {true_coords.shape}')
        # print(f'pred first: {pred_coords[1, 900, 85:90]}')
        # print(f'true first: {true_coords[1, 900, 85:90]}')

        # Apply the mask to both predicted and true coordinates
        pred_coords_masked = pred_coords[mask]
        true_coords_masked = true_coords[mask]

        # print(f'pred second: {pred_coords[1, 900, 85:90]}')
        # print(f'true second: {true_coords[1, 900, 85:90]}')
        # print()


        # Calculate RMSD only on the masked coordinates
        diff = pred_coords_masked - true_coords_masked
        rmsd_value = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))
        return rmsd_value



def train_model(model,
                dataset,
                criterion,
                optimizer,
                epochs=10,
                batch_size=2,
                shuffle=True,
                device='cpu',
                print_interval=10,
                # Add gradient accumulation parameters
                accumulation_steps=1, # Accumulate gradients over this many steps
                save_after=1000,
                save_loc=None):
    """
    Train a given model on the provided dataset.

    Args:
        model (nn.Module): The PyTorch model to train.
        dataset (Dataset): A PyTorch Dataset instance providing (input, target) pairs.
        criterion (nn.Module): The loss function to use.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        epochs (int): Number of epochs to train.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset before each epoch.
        device (str or torch.device): Device on which to do the training ('cpu' or 'cuda').
        print_interval (int): Print the loss every `print_interval` iterations.
        accumulation_steps (int): Number of steps to accumulate gradients before updating.

    Returns:
        model (nn.Module): The trained model.
        history (dict): A dictionary containing training loss history.
    """
    m = 0 #Tracks the save state of our model!
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate_fn_two)
    dataset.set_batch_size(batch_size)
    model.train()
    model.to(device)

    history = {"loss": []}
    iteration = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (seq_emb, coords_true) in enumerate(dataloader):
            # print(f'something returned')
            # print(f'seq_emb shape: {seq_emb.shape}')
            # print(f'coord_true sha: {coords_true.shape}')
            # print(f'coord_true: {coords_true[5, 50:55, 87]}')
            seq_emb = seq_emb.to(device)  # [B, L, D]
            coords_true = coords_true.to(device)  # [B, L, 3]

            # Forward pass
            coords_pred = model(seq_emb)
            loss = criterion(coords_pred, coords_true)

            # Scale the loss by accumulation steps
            loss = loss / accumulation_steps

            # Backward pass and accumulate gradients
            loss.backward()

            # Update parameters every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            iteration += 1

            # Print loss & save model at given intervals
            if iteration % save_after == 0:
                logger.info(f'saved model {m}')
                logger.info(f'Epoch {epoch+1}/{epochs}, Iteration {iteration}, Loss: {loss.item():.4f}')
                torch.save(model, f'{save_loc}-{m}')
                m += 1
            if iteration % print_interval == 0:
                print(f"Epoch {epoch+1}/{epochs}, Iteration {iteration}, Loss: {loss.item():.4f}")

        # Average loss for the epoch
        epoch_loss /= len(dataloader)
        history["loss"].append(epoch_loss)
        print(f"Epoch {epoch+1} completed. Average Loss: {epoch_loss:.4f}")

    return model, history



def unstack(coords):
    """
    Unstacks the coordinates
    """
    N, M = coords.shape
    N = N * 27
    M = M // 27
    coords = coords.detach().numpy()
    # reshaped_parts = coords.reshape(N // 27, 27, M)
    # original_array = reshaped_parts.transpose(0, 2, 1).reshape(N, M)

    reshaped_parts = coords.reshape(N // 27, 27, M)
    # original_array = reshaped_parts.transpose(0, 2, 1).reshape(N, M)

    # How we stack:
    # reshaped_target = target.reshape(N // 27, 27, M)  # Split into groups of 27 rows
    # stacked_target = reshaped_target.reshape(N // 27, M * 27)
    # original_array = reshaped_parts.transpose(0, 2, 1).reshape(N, M)
    # return original_array
    return reshaped_parts








if __name__ == "__main__":
    # System arguments: Node name, reprocess, data size, num_heads, depth!
    if len(sys.argv) > 1:
        node_name = sys.argv[1]
        reprocess = sys.argv[2]
        try:
            data_size = int(sys.argv[3])
        except:
            data_size = 10
        try:
            num_heads = int(sys.argv[4])
            depth = int(sys.argv[5])
        except:
            num_heads = 8
            depth = 4

    else:
        node_name = 'Default'
        reprocess = 't'
        data_size = 10
        num_heads = 8
        depth = 4

    setup(node_name=node_name)


    # ---------------------- Logging framework ----------------------
    # 10MB handlers
    file_handler = logging.handlers.RotatingFileHandler(f'Logs/{node_name}/Full_Log.log', maxBytes=10000000,
                                                        backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    # Starts each call as a new log!
    file_handler.doRollover()

    master_handler = logging.FileHandler(f'Logs/{node_name}/ERRORS.log', mode='w')
    master_handler.setLevel(logging.WARNING)

    logging.basicConfig(level=logging.DEBUG, handlers=[master_handler, file_handler],
                        format='%(levelname)-8s: %(asctime)-22s %(module)-20s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S | ')

    multiprocessing_logging.install_mp_handler()

    logger.info(f'Started with following system variables:')
    logger.info(f'{sys.argv}')
    logger.info(f'node_name: {node_name}')
    logger.info(f'reprocess: {reprocess}')
    logger.info(f'num_heads: {num_heads}')
    logger.info(f'depth: {depth}')


    print(f'node_name: {node_name}')
    print(f'reprocess: {reprocess}')
    print(f'num_heads: {num_heads}')
    print(f'depth:     {depth}')

    # ---------------------- End Logging Framework ----------------------



    experiment_number = 0
    f = open('trial_tracker.txt', 'r+')
    attempt_num = int(f.readline())

    f = open('trial_tracker.txt', 'r+')
    f.writelines(f'{attempt_num + 1}\n')
    f.close()



    start = time.time()
    if reprocess.lower() == 't':
        logger.warning(f' ------------------------- Beginning Parsing Sequences ------------------------- ')
        a = parse_seq.Sequence_Parser(max_samples=data_size)
        # a.parse_names(['6XTB'])
        print(a.e.encode)
        a.RAM_Efficient_parsing(batch_size=10)
        # a.open_struct('6XTB')

        # logging.info(f'Took {time.time() - start} seconds!!!')
        logger.warning(f'Complete! Took {time.time() - start} seconds!!!')

        logger.warning(' --------------------------------- Begin Processing Data ---------------------------------------- ')

        process_data(max_proteins=data_size)

    logger.warning(' --------------------------------- Begin Transformer ---------------------------------------- ')
    output_len = 2430

    train_seq = np.load('PDBs/big_data/in-train.npy', mmap_mode='r', allow_pickle=True)
    train_tar = np.load('PDBs/big_data/out-train.npy', mmap_mode='r', allow_pickle=True)

    print(f'train_seq: {train_seq.shape}')


    esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device)

    pdb_dir = "path_to_pdbs"
    dataset = ProteinStructureDataset(pdb_dir, esm_model, esm_batch_converter, train_seq, train_tar, device, num_training=int(data_size * 0.8))
    model = ProteinStructurePredictor(embed_dim=esm_model.embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.0)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = RMSDLoss()

    # # seq = 'N'
    # seq = 'AFLAAAYGA'
    # # AFLAAAYGA
    # # tokens: tensor([[ 0,  5, 18,  4,  5,  5,  5, 19,  6,  5,  2]])
    # # N
    # # tokens: tokens: tensor([[ 0, 17,  2]])
    #
    # a = [ord(n) - 60 for n in seq]
    # print(a)
    #
    # batch = [("protein", seq)]
    # _, _, tokens = esm_batch_converter(batch)
    # print(f'tokens: {tokens}')
    #
    # tokens = tokens.to(next(esm_model.parameters()).device)
    # with torch.no_grad():
    #     results = esm_model(tokens, repr_layers=[esm_model.num_layers])
    # # Exclude CLS token
    # esm_emb_b = results["representations"][esm_model.num_layers][0, 1:len(seq) + 1, :]




    # train_seq = np.memmap('PDBs/big_data/in-train.bin')
    # train_tar = np.memmap('PDBs/big_data/out-train.bin', shape=(-1, 2430))

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # print(f'seq shape: {train_seq.shape}')
    # print(f'tar shape: {train_tar.shape}')
    #
    # x, t = get_batch(train_seq, train_tar, 100, 3, device)
    #
    # dataset.to_embedding(x)

    print(f'cuda available? {torch.cuda.is_available()}')
    print(f'torch version: {torch.version.cuda}')
    print(f'torch device: {device}')



    train_model(model, dataset, criterion, optimizer, epochs=10, batch_size=2, shuffle=True, device=device,
                print_interval=50, save_after=100, save_loc=f'models/{node_name}/Save')

    # Run a single example to evaluate our predictions!
    # Just make sure it produces something reasonable.
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)
    model.eval()
    model.to(device)

    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_fn_two)
    for batch_idx, (seq_emb, coords_true) in enumerate(dataloader):
        coords_true = coords_true.to(device)
        seq_emb = seq_emb.to(device)
        coords_pred = model(seq_emb)
        break

    print(f'seq_emb: {seq_emb.shape}')

    # seq_emb = seq_emb.to(device)  # [B, L, D]
    #             coords_true = coords_true.to(device)  # [B, L, 3]

    #             # Forward pass
    #             coords_pred = model(seq_emb)

    # Display the result of our single prediction
    # print(f'coords true: {coords_true}')
    print(f'original shape: {coords_true.shape}')
    print(f'original pred shape: {coords_pred.shape}')

    print(f'selection: {coords_true[0, 50:55, 87]}')

    ct_cpu = coords_true.to('cpu')[0]
    cp_cpu = coords_pred.to('cpu')[0]


    # ct_cpu = coords_true.cpu()
    # cp_cpu = coords_pred.cpu()

    unstacked_true = unstack(ct_cpu)
    unstacked_pred = unstack(cp_cpu)
    print(f'final shape     : {unstacked_true.shape}')
    print(f'prediction shape: {unstacked_pred.shape}')
    # print(f'true: {unstacked_true.shape}')

    # print(f'true: {unstacked_true[3, :]}')
    # print(f'unstacked true: {unstacked_true[:100, -5:]}')
    # print(f'unstacked pred: {unstacked_pred[:100, -5:]}')

    # for t, p in zip(unstacked_true[:100, -5:], unstacked_pred[:100, -5:]):
    #     print(f'true: {t}')
    #     print(f'pred: {p}')
    #     print()

    # for t, p in zip(unstacked_true[50:100, -5:], unstacked_pred[50:100, -5:]):
    #     print(f'true: {t}')
    #     print(f'pred: {p}')
    #     print()

    for i in range(5):
        for t, p in zip(unstacked_true[i, :, -5:], unstacked_pred[i, :, -5:]):
            print(f'true: {t}')
            print(f'pred: {p}')
            print()
        print()


    logger.info(f'Complete!!! Took {time.time() - start} seconds!!')
    torch.save(model, f'models/model-FINISHED-{node_name}-{attempt_num}')
    logger.info(f'Saved model successfully!')

    print(f'----------------------------------------------------------------------------------------------')




    # Example code for if we want to turn off the gradient of our undefined input
    # def get_hook(param_idx):
    #  def hook(grad):
    #       grad = grad.clone() # NEVER change the given grad inplace
    #       # Assumes 1D but can be generalized
    #       for i in grad.size(0):
    #           if should_be_disabled[param_idx][i]:
    #               grad[i] = 0
    #       return grad
    #   return hook


