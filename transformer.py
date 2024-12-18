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
from logging_setup import get_logger, setup_logger, change_log_code
import time
import parse_seq
import sys
# random.seed(777)


esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
esm_batch_converter = esm_alphabet.get_batch_converter()

block_size = 1000



def setup(node_name=None):
    """
    Creates directories to store all files
    :param node_name:
    :return:
    """
    os.makedirs(f'PDBs', exist_ok=True)
    os.makedirs(f'PDBs/all', exist_ok=True)
    os.makedirs(f'PDBs/big_data', exist_ok=True)
    os.makedirs(f'PDBs/pre_processed_data', exist_ok=True)
    os.makedirs(f'PDBs/Residues', exist_ok=True)
    os.makedirs(f'PDBs/accuracies', exist_ok=True)

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

    DOES NOT re-scale.

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
    centered_coords[valid_mask, :] -= mean_defined[:]

    # Create the centered array
    centered_array = np.hstack((target[:, :-5], centered_coords, target[:, -2:]))

    return centered_array



class protein_unifier():
    """
    "Unifies" the proteins and appends into a single file.
    Stacks proteins on top of each other onto a single array.

    Stores array in storage with a memmap to avoid excess memory usage. Allows very large datasets
    with many proteins, because dataset is NOT held in memory!

    """
    def __init__(self, num_looking, name='train'):
        # Inform what set we're on
        logger.debug(f'--- saving {name} set ---')
        # in & out files will be memmaps
        self.in_file = None
        self.out_file = None
        self.save_path = 'PDBs/big_data/'
        # num_looking helps us initialize our array in storage to be the correct size.
        self.num_looking = num_looking
        # tracks the current protein number. Helps us know where in our array to update.
        self.track = 0
        self.name = name

    def tokenize(self, sequence):
        """
        Turns an amino letter sequence into tokens for ESM embedding!
        Tokens are ints. First and final tokens are <BOS> and <EOS> tokens.

        * IMPORTANT NOTE * This is case-sensitive!! BOS, EOS, and PAD are lowercase letters, while aminos are uppercase.

        :param sequence: String list of amino acid letters
        :return: Int array of tokenized sequence
        """
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

        return sequence


    def add_protein(self, sequence, protein):
        """
        Adds a protein to our memmap!

        :param sequence: String of our 1-letter amino acid sequence.
        :param protein: batch_size * 2430 array storing target protein
        :return: None
        """

        # Initialize our out_file if not already created
        if self.out_file is None:
            self.out_file = np.memmap(f'{self.save_path}out-{self.name}', dtype='float32', mode='w+', shape=(protein.shape[0] * self.num_looking, protein.shape[1]))
            self.out_file[:protein.shape[0], :] = protein

        else:
            # Otherwise, add protein to our memmap.
            self.out_file[(self.track) * protein.shape[0]:(self.track + 1) * protein.shape[0], :] = protein

        # turn sequence into correct tokens before saving
        sequence = self.tokenize(sequence)

        # Initialize our in_file if not already created
        if self.in_file is None:
            self.in_file = np.memmap(f'{self.save_path}in-{self.name}', dtype='int', mode='w+',
                                      shape=(sequence.shape[0] * self.num_looking,))
            self.in_file[:sequence.shape[0]] = sequence
        else:
            # Add our sequence to its correct place in the array.
            self.in_file[(self.track) * sequence.shape[0]:(self.track + 1) * sequence.shape[0]] = sequence


        # update our iteration tracker
        self.track += 1

        # Track our progress
        progress = round(((self.track / self.num_looking) * 100), 2)
        logger.debug(f'Successfully Processed! {progress}')

        # Double check our protein shape and sequence shape are in agreement!
        if protein.shape[0] != sequence.shape[0]:
            logging.error(f'Sequence & Protein Misaligned!! Location {self.track}')
            logging.info(f'protein shape: {protein.shape}')
            logging.info(f'sequence shape: {sequence.shape}')


    def save(self, name):
        """
        Save the memmaps as their own npy objects!
        :param name: Test/train/target. String that we want to call our numpy array.
        :return: None
        """
        # No longer looking at a specific protein, remove protein code from logging
        change_log_code()

        # save both our in file and out file
        np.save(f'{self.save_path}in-{name}', allow_pickle=True, arr=self.in_file)
        np.save(f'{self.save_path}out-{name}', allow_pickle=True, arr=self.out_file)

        # Close our memmap files!
        del self.out_file
        del self.in_file

    def __repr__(self):
        msg = 'PROTEIN UNIFIER: \n' + f'total accumulated sequence length: {len(self.in_file)} \n' + f'total accumulated protein  length: {self.out_file.shape} \n'
        return msg



def stack_atoms(target):
    """
    Stacks 27-dimensional arrays into single long array.

    Run this BEFORE running format_target!
    :param target: Array for our protein. (N by 90) where N is the number of atoms. Should be a multiple of 27.
    :return: Array for our protein. Shape is (n, 2430) = (N // 27, 90 * 27)
    """
    # Target.shape[0] must be multiple of 27.
    if target.shape[0] % 27 != 0:
        logging.warning(f'Invalid protein shape for stacking. {target.shape}')

    # stack groups of 27 atoms into a single row!
    N, M = target.shape
    reshaped_target = target.reshape(N // 27, 27, M)  # Split into groups of 27 rows
    stacked_target = reshaped_target.reshape(N // 27, M * 27)
    return stacked_target


def format_target(target):
    """ Formats a single protein target label.

    Adds BOS & EOS rows to beginning of sequence and end of sequence.
    Rows are filled entirely with 0s except for the identifier.
    - BOS identifier is index 82
    - EOS identifier is index 83
    - PAD identifier is index 84
    Add as many rows with PAD as necessary to make `target` correct sizing.

    *NOTE* Size of proteins is block_size - 2! Block_size is final size, we add BOS & EOS.
    """
    sequence_length = block_size

    # BOS and EOS row are added to stacked arrays.
    BOS_row = np.zeros(target.shape[1], dtype=int)
    EOS_row = np.zeros(target.shape[1], dtype=int)
    BOS_row[82] = 1
    EOS_row[83] = 1

    # Add 2 because we're adding BOS and EOS rows!
    # If the target isn't as long as our desired sequence length...
    if target.shape[0] + 2 < sequence_length:
        # fill with padding.
        PAD = np.zeros((sequence_length - target.shape[0] - 2, target.shape[1]))
        # Make sure our padding rows are flagged.
        PAD[:, 84] = 1
        # t is our new target
        # Stack the BOS & EOS to the top and bottom of our target.
        t = np.vstack([BOS_row, target[:sequence_length - 2], PAD, EOS_row])
    else:
        t = np.vstack([BOS_row, target[:sequence_length - 2], EOS_row])

    return t


def format_input(target):
    """ Formats a single sample (given, target)

    Adds BOS & EOS rows to beginning of sequence and end of sequence.
    Rows are filled entirely with 0s except for the identifier.
    - BOS identifier is index 82
    - EOS identifier is index 83
    - PAD identifier is index 84
    Add as many rows with PAD as necessary to make `target` correct sizing.

    *NOTE* Size of sequence is block_size - 2! Block_size is final size, we add BOS & EOS.
    """
    sequence_length = block_size

    # Use special case-sensitive letters to denote BOS, EOS, and PAD
    BOS = 'b'
    EOS = 'e'
    PAD = 'p'

    # Make sure our target is uppercase so that our lowercase BOS, EOS, and PAD can be differentiated.
    target = str(target).upper()
    # If our sequence is longer than our target length...
    if len(target) > sequence_length + 2:
        logger.warning(f'Sequence size exceeds batch size. Sequence size: {len(target)}')

    if len(target) + 2 < sequence_length:
        padding = PAD * (sequence_length - len(target) - 2)
        # t is our new target
        t = BOS + target[:sequence_length - 2] + padding + EOS
    else:
        t = BOS + target[:sequence_length - 2] + EOS


    return t




def process_data(max_proteins=1000):
    """
    Processes all proteins in the open_dir directory.

    Saves these processed proteins to the save_dir.

    Does NOT take these proteins as input so that we can perform pre-processing once and save our results.
    :param max_proteins: Maximum number of proteins we will process. *NOTE* Random which proteins are ommitted (if any)
    :return: None
    """
    # Where to obtain pre-processed proteins
    open_dir = 'PDBs/pre_processed_data'
    # Where to save our post-processed proteins
    save_dir = 'PDBs/processed_data'

    os.makedirs(save_dir, exist_ok=True)

    # Get set of protein codes we have already pre-processed
    code_set = {name[:4] for name in os.listdir(open_dir)}
    code_set = list(code_set)
    # Shuffles code set in place.
    random.shuffle(code_set)
    # cut off excess proteins
    code_set = code_set[:max_proteins]

    logger.info(f'all names: {code_set}')

    # make our train-test split
    train_codes = code_set[:int(len(code_set) * 0.8)]
    valid_codes = code_set[int(len(code_set) * 0.8):int(len(code_set) * 0.9)]
    test_codes = code_set[int(len(code_set) * 0.9):]



    # Saves processed data into save_dir under test, train, and valid

    pu = protein_unifier(len(train_codes), name='train')
    for code in train_codes:
        # Change our logger to reflect the protein code
        change_log_code(code)
        try:
            # Open our pre-processed data
            given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
            target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

            # process data
            g = format_input(given)
            # turn atom name into one-hot encoding...
            onehot_target = one_hot_encode_column(target, 0, 85)
            # normalize our coordinates
            onehot_target = normalize_target(onehot_target)
            # stack our atoms on top of each other
            onehot_target = stack_atoms(onehot_target)
            # Add padding to entire sequence
            onehot_target = format_target(onehot_target)

            pu.add_protein(g, onehot_target)
        except:
            logger.exception(f'FATAL ERROR WITH THIS PROTEIN! {code}')
    pu.save('train')


    # Move on to validation set
    pu = protein_unifier(len(valid_codes), name='valid')
    for code in valid_codes:
        # Change our logger to reflect the protein code
        change_log_code(code)
        try:

            given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
            target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

            # process data
            g = format_input(given)
            # turn atom name into one-hot encoding...
            onehot_target = one_hot_encode_column(target, 0, 85)
            # normalize our coordinates
            onehot_target = normalize_target(onehot_target)
            # stack our atoms on top of each other
            onehot_target = stack_atoms(onehot_target)
            # Add padding to entire sequence
            onehot_target = format_target(onehot_target)

            pu.add_protein(g, onehot_target)
        except:
            logger.exception(f'FATAL ERROR WITH THIS PROTEIN! {code}')

    pu.save('valid')

    # Move on to test set
    pu = protein_unifier(len(test_codes), name='test')
    for code in test_codes:
        # Change our logger to reflect the protein code
        change_log_code(code)
        try:

            given = np.load(f'{open_dir}/{code}-in.npy', mmap_mode='r', allow_pickle=True)
            target = np.load(f'{open_dir}/{code}-target.npy', mmap_mode='r', allow_pickle=True)

            print(f'given: {given}')

            # process data
            g = format_input(given)
            # turn atom name into one-hot encoding...
            onehot_target = one_hot_encode_column(target, 0, 85)
            # normalize our coordinates
            onehot_target = normalize_target(onehot_target)
            # stack our atoms on top of each other
            onehot_target = stack_atoms(onehot_target)
            # Add padding to entire sequence
            onehot_target = format_target(onehot_target)

            pu.add_protein(g, onehot_target)
        except:
            logger.exception(f'FATAL ERROR WITH THIS PROTEIN! {code}')

    pu.save('test')
    change_log_code()



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




def custom_collate_fn_two(batch):
    """
    Converts batch with list of (embedding, coordinate) pairs into valid batch tensor with shape (B, L, D)
    B is batch size
    L is block size (sequence length)
    D is embedding size (320 for esm embedding)

    """

    # pick arbitrary batch element and use shape of embedding
    # then use same  batch element and use shape of target
    x = torch.zeros(len(batch), batch[0][0].shape[1], batch[0][0].shape[2])
    t = torch.zeros(len(batch), batch[0][1].shape[0], batch[0][1].shape[1])

    # pack each element in our batch into a single tensor
    for i, (esm_emb, coords) in enumerate(batch):
        x[i, :, :] = esm_emb
        t[i, :, :] = coords


    if device == 'cuda':
        # pin arrays x,t, which allows us to move them to GPU asynchronously
        #  (non_blocking=True)
        x, t = x.pin_memory().to(device, non_blocking=True), t.pin_memory().to(device, non_blocking=True)
    else:
        x, t = x.to(device), t.to(device)
    return x, t




class ProteinStructureDataset(Dataset):
    """
    Handles our dataset!
    """
    def __init__(self, esm_model, esm_batch_converter, train_seq, train_tar, device, num_training):
        # ESM model gets us our embeddings from our tokenized sequence
        self.esm_model = esm_model
        self.esm_batch_converter = None
        self.train_seq = train_seq
        self.train_tar = train_tar

        self.device = device
        self.traversed = 0

        self.starts = np.where(self.train_seq == 0)[0]
        self.num_training = num_training

        logger.info(f'Training size: {num_training}')
        print(f'Training size: {num_training}')


    def __len__(self):
        return self.num_training

    def __iter__(self):
        return self


    def __getitem__(self, idx):
        """
        Returns a tuple with our (embedding, target coordinates).

        Our Dataloader calls this many times and is used by our custom collate function.
        :param idx: index to pull from our training target!
        :return:
        """
        coords = torch.from_numpy((self.train_tar[idx:idx + block_size]))
        seq = torch.from_numpy((self.train_seq[idx:idx + block_size]))
        seq = seq[None, :]
        esm_emb = self.to_embedding(seq)

        return esm_emb, coords



    def to_embedding(self, tokens):
        """
        Converts a batch of tokens to a proper embedding!
        # Input: (B, L, 1)
        # Output: (B, L, Embedding_size)

        :param indices:
        :return:
        """
        # Send tokens to correct device
        tokens = tokens.to(next(self.esm_model.parameters()).device)
        # No gradient when getting these results
        with torch.no_grad():
            results = self.esm_model(tokens, repr_layers=[self.esm_model.num_layers])
        esm_emb = results["representations"][self.esm_model.num_layers]

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
        # # Create a mask to ignore padded regions
        # mask = (true_coords.sum(dim=-1) != 0)  # Assuming padded coords are all zeros
        # # mask_two = (true_coords[:, :, 83] != 0)
        #
        # # mask = (true_coords[:, :, 83] != 0)
        #
        # print(f'mask shape: {mask.shape}')
        # # # mask shape: torch.Size([10, 1000])
        # # print(f'mask: {mask}')
        # # print(f'mask_two: {mask_two}')
        # # print(f'mask_two shape: {mask_two.shape}')
        # # print(f'true coords shape: {true_coords.shape}')
        # # false_indices = torch.where(mask_two == False)[1]
        # #
        # # print(false_indices[840:900])
        # # print(f'false indices shape: {false_indices.shape}')
        #
        # # torch.Size([2, 463, 2430])
        # # pred_coords has size (batch, block, output)
        # # print(f'pred coords: {pred_coords.shape}')
        # # print(f'true coords: {true_coords.shape}')
        # # print(f'pred first: {pred_coords[1, 900, 85:90]}')
        # # print(f'true first: {true_coords[1, 900, 85:90]}')
        #
        # # Apply the mask to both predicted and true coordinates
        # pred_coords_masked = pred_coords[mask]
        # true_coords_masked = true_coords[mask]
        # print(f'pred coords masked: {pred_coords_masked.shape}')
        # print(f'pred coords: {pred_coords.shape}')
        # diff = pred_coords_masked - true_coords_masked
        # rmsd_value = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))
        # print(f'rmsd: {rmsd_value}')
        # print(f'rmsd shape: {rmsd_value.shape}')

        mask = true_coords[:, :, 83] != 1  # shape: (Batch, Sequence Length)

        # Apply the mask to both predicted and true coordinates
        # We will use the mask to keep the rows where dimension 83 is not 1
        pred_coords_masked = pred_coords[mask]
        true_coords_masked = true_coords[mask]

        # Compute the squared differences
        diff = pred_coords_masked - true_coords_masked  # shape: (valid points, Dimension)
        squared_diff = diff ** 2  # element-wise squared differences

        # Sum the squared differences across the dimensions (axis=-1)
        squared_diff_sum = squared_diff.sum(dim=-1)  # sum over the dimension axis

        # Sum across the batch and sequence length axis
        total_squared_diff = squared_diff_sum.sum()  # sum over all points

        # Get the number of valid points
        num_valid_points = mask.sum().item()

        # Compute the RMSD
        rmsd = torch.sqrt(total_squared_diff / num_valid_points)

        return rmsd






        # # Updated:
        # # pred coords masked: torch.Size([8, 2430])
        # # pred coords: torch.Size([8, 1000, 2430])
        # # mask shape: torch.Size([8, 1000])
        #
        # # Old:
        # # pred coords masked: torch.Size([8000, 2430])
        # # pred coords: torch.Size([8, 1000, 2430])
        # # mask shape: torch.Size([8, 1000])
        #
        # # print(f'pred second: {pred_coords[1, 900, 85:90]}')
        # # print(f'true second: {true_coords[1, 900, 85:90]}')
        # # print()
        #
        #
        # # Calculate RMSD only on the masked coordinates
        # diff = pred_coords_masked - true_coords_masked
        # rmsd_value = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=-1)))
        # return rmsd_value



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
    model.train()
    model.to(device)

    history = {"loss": []}
    iteration = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (seq_emb, coords_true) in enumerate(dataloader):
            # Move samples to graphics card (or cpu)
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
    """ Puts each atom back into its own row.

    Unstacks the coordinates from (N, 2430) to (M, 90).
    """
    N, M = coords.shape
    N = N * 27
    M = M // 27

    coords = coords.detach().numpy()
    reshaped_parts = coords.reshape(N // 27, 27, M)

    return reshaped_parts



def process_sys_args(args):
    """
    Processes system arguments and handles gracefully when not correct
    :param args:
    :return:
    """
    if len(args) > 1:
        node_name = args[1]
        reprocess = args[2]
        try:
            data_size = int(args[3])
        except:
            data_size = 10
        try:
            num_heads = int(args[4])
        except:
            num_heads = 8

        try:
            depth = int(args[5])
        except:
            depth = 4

        try:
            batch_size = int(args[6])
        except:
            batch_size = 10

        try:
            num_epochs = int(args[7])
        except:
            num_epochs = 10

    else:
        node_name = 'Default'
        reprocess = 't'
        data_size = 4
        num_heads = 8
        depth = 4
        batch_size = 2
        num_epochs = 100

    return node_name, reprocess, data_size, num_heads, depth, batch_size, num_epochs





if __name__ == "__main__":
    # System arguments: Node name, reprocess, data size, num_heads, depth, batch_size, num_epochs!
    node_name, reprocess, data_size, num_heads, depth, batch_size, num_epochs = process_sys_args(sys.argv)

    # Do our setup
    setup(node_name=node_name)
    logger = setup_logger(node_name=node_name)

    # Note run info for our logger
    logger.info(f'Started with following system variables:')
    logger.info(f'{sys.argv}')
    logger.info(f'node_name:  {node_name}')
    logger.info(f'reprocess:  {reprocess}')
    logger.info(f'num_heads:  {num_heads}')
    logger.info(f'depth:      {depth}')
    logger.info(f'batch_size: {batch_size}')
    logger.info(f'num_epochs: {num_epochs}')


    print(f'node_name: {node_name}')
    print(f'reprocess: {reprocess}')
    print(f'num_heads: {num_heads}')
    print(f'depth:     {depth}')

    # Track our experiment number
    experiment_number = 0
    f = open('trial_tracker.txt', 'r+')
    attempt_num = int(f.readline())

    f = open('trial_tracker.txt', 'r+')
    f.writelines(f'{attempt_num + 1}\n')
    f.close()


    # Parse sequences if we're asked to
    start = time.time()
    if reprocess.lower() == 't':
        logger.info(f'------------------------- Beginning Parsing Sequences ------------------------- ')
        a = parse_seq.Sequence_Parser(max_samples=data_size)
        print(a.e.encode)
        a.RAM_Efficient_parsing(batch_size=10)


        logger.info(f'Complete! Took {time.time() - start} seconds!!!')
        # Process data if we're asked to
        logger.info('--------------------------------- Begin Processing Data ---------------------------------------- ')

        process_data(max_proteins=data_size)

    logger.info('--------------------------------- Begin Transformer ---------------------------------------- ')
    output_len = 2430

    # Load our processed training and target sequences
    train_seq = np.load('PDBs/big_data/in-train.npy', mmap_mode='r', allow_pickle=True)
    train_tar = np.load('PDBs/big_data/out-train.npy', mmap_mode='r', allow_pickle=True)

    # Load ESM-Fold for creating our embeddings
    esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model.eval()

    # Send model to graphics card
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model = esm_model.to(device)

    # Initialize our model and dataset
    dataset = ProteinStructureDataset(esm_model, esm_batch_converter, train_seq, train_tar, device, num_training=int(data_size * 0.8))
    model = ProteinStructurePredictor(embed_dim=esm_model.embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=4.0)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = RMSDLoss()

    # CUDA info
    logger.info(f'- cuda info -')
    logger.info(f'cuda available? {torch.cuda.is_available()}')
    logger.info(f'torch version: {torch.version.cuda}')
    logger.info(f'torch device: {device}')
    print(f'cuda available? {torch.cuda.is_available()}')
    print(f'torch version: {torch.version.cuda}')
    print(f'torch device: {device}')

    # Train our model!!
    train_model(model, dataset, criterion, optimizer, epochs=num_epochs, batch_size=batch_size, shuffle=True, device=device,
                print_interval=50, save_after=100, save_loc=f'models/{node_name}/Save')


    # --------------------------------------------------------------------------------------------------------------------------------
    # Run a single example to evaluate our predictions!
    # Just make sure it produces something reasonable.
    model.eval()
    model.to(device)

    # Pull out a single example
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=custom_collate_fn_two)
    for batch_idx, (seq_emb, coords_true) in enumerate(dataloader):
        coords_true = coords_true.to(device)
        seq_emb = seq_emb.to(device)
        coords_pred = model(seq_emb)
        break



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

    print(f'unstacked shape: {unstacked_true.shape}')
    # unstacked has shape (sequence_length, atoms, atom dimensions)

    for i in range(5):
        # do 50 + i to get to the interesting amino acids
        for t, p in zip(unstacked_true[50 + i, :, -5:], unstacked_pred[50+i, :, -5:]):
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


