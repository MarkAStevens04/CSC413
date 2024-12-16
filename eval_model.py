import torch
import os
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
import time
import sys
import seaborn as sns
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


block_size = 1000
os.makedirs(f'PDBs/accuracies', exist_ok=True)
os.makedirs(f'PDBs/accuracies/train_acc', exist_ok=True)
os.makedirs(f'PDBs/accuracies/valid_acc', exist_ok=True)
os.makedirs(f'PDBs/accuracies/train_loss', exist_ok=True)
os.makedirs(f'PDBs/accuracies/valid_loss', exist_ok=True)
os.makedirs(f'PDBs/accuracies/train_class', exist_ok=True)
os.makedirs(f'PDBs/accuracies/valid_class', exist_ok=True)

def open_model(model_dir):
    with torch.no_grad():
      model = torch.load(model_dir, weights_only=False)
      return model



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
    # print(f'len batch: {len(batch)}')
    x = torch.zeros(len(batch), batch[0][0].shape[1], batch[0][0].shape[2])
    t = torch.zeros(len(batch), batch[0][1].shape[0], batch[0][1].shape[1])
    # print(f'x shape: {x.shape}')
    # print(f't shape: {t.shape}')

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


  def __getitem__(self, idx):

      coords = torch.from_numpy((self.train_tar[idx:idx + block_size]))
      seq = torch.from_numpy((self.train_seq[idx:idx + block_size]))
      seq = seq[None, :]
      esm_emb = self.to_embedding(seq)


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
      starts = self.batches[traversed]
      ix = self.starts[starts]

      x = torch.stack([torch.from_numpy((seq[i:i + block_size])) for i in ix])
      t = torch.stack([torch.from_numpy((tar[i:i + block_size, :])) for i in ix])
      if device == 'cuda':
          # pin arrays x,t, which allows us to move them to GPU asynchronously
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




def unstack(coords):
    """
    Unstacks the coordinates
    """
    B, N, M = coords.shape
    N = N * 27
    M = M // 27
    coords = coords.detach().numpy()
    # reshaped_parts = coords.reshape(N // 27, 27, M)
    # original_array = reshaped_parts.transpose(0, 2, 1).reshape(N, M)

    reshaped_parts = coords.reshape((N // 27) * B, 27, M)
    # original_array = reshaped_parts.transpose(0, 2, 1).reshape(N, M)

    # How we stack:
    # reshaped_target = target.reshape(N // 27, 27, M)  # Split into groups of 27 rows
    # stacked_target = reshaped_target.reshape(N // 27, M * 27)
    # original_array = reshaped_parts.transpose(0, 2, 1).reshape(N, M)
    # return original_array
    return reshaped_parts



def calculate_tm_score(true_coords, pred_coords, d0=None):
    """
    Calculate TM-score for structural alignment of unstacked coordinates

    Parameters:
    - true_coords: numpy array of true coordinates
    - pred_coords: numpy array of predicted coordinates
    - d0: distance cutoff (if None, calculated automatically)

    Returns:
    - TM-score (float between 0 and 1)
    """
    # Ensure inputs are numpy arrays
    true_coords = np.array(true_coords)
    pred_coords = np.array(pred_coords)
    # print(f'true coords shape: {true_coords.shape}')
    # print(f'first 3: {true_coords[0, 1, -5:]} {true_coords[0, 1, 1]}')

    # get the lists of coordinates only for atoms that have a defined position
    known_aminos = np.argwhere(true_coords[:, :, -2] == 1)

    # known_true = true_coords[known_aminos]
    # known_pred = pred_coords[known_aminos]
    known_true = true_coords[known_aminos[:, 0], known_aminos[:, 1], -5:-2]
    known_pred = pred_coords[known_aminos[:, 0], known_aminos[:, 1], -5:-2]
    # print(f'true coords: {true_coords.shape}')
    # for i in known_true:
    #     print(f'true: {i}')
    #     print(f'pred: {i}')
    # print(f'known aminos: {known_aminos.shape}')
    # print(f'some known aminos: {known_aminos}')
    # for i in known_aminos:
    #     print(f'known: {true_coords[i[0], i[1], -5:-2]} {true_coords[i[0], i[1], 1]}')
    #     print(f'pred : {pred_coords[i[0], i[1], -5:-2]} {pred_coords[i[0], i[1], 1]}')

    # print(f'known true: {known_true.shape}')
    # print(f'known pred: {known_pred.shape}')

    # for i, true, pred in zip(known_aminos, known_true, known_pred):
    #   print(f'i: {i}')
    #   print(f'true: {true_coords[i[0], i[1], -5:]}')
    #   print(f't   : {true}')
    #   print(f'pred: {pred_coords[i[0], i[1], -5:]}')
    #   print(f'p   : {pred}')


    # Check input shapes
    if known_true.shape != known_pred.shape:
        raise ValueError("True and predicted coordinates must have the same shape")

    # Calculate number of points
    L = known_true.shape[0]

    # Calculate pairwise distances
    distances = np.sqrt(np.sum((known_true - known_pred)**2, axis=1))

    # Calculate d0 if not provided
    if d0 is None:
        # Standard TM-score d0 calculation
        d0 = 1.24 * np.power(L - 15, 1/3) - 1.8

    # Calculate TM-score
    tm_scores = 1 / (1 + (distances / d0)**2)

    # Average TM-score
    tm_score = np.mean(tm_scores)

    return tm_score

def calculate_gdt_ts(true_coords, pred_coords, distance_thresholds=[1, 2, 4, 8]):
    """
    Calculate GDT_TS (Global Distance Test Total Score)

    Parameters:
    - true_coords: numpy array of true coordinate points
    - pred_coords: numpy array of predicted coordinate points
    - distance_thresholds: list of distance thresholds to evaluate (default: [1, 2, 4, 8])

    Returns:
    - GDT_TS score (float between 0 and 100)
    - Detailed breakdown of points within each threshold
    """
    # Ensure inputs are numpy arrays
    true_coords = np.array(true_coords)
    pred_coords = np.array(pred_coords)



    # get the lists of coordinates only for atoms that have a defined position
    known_aminos = np.argwhere(true_coords[:, :, -2] == 1)

    known_true = true_coords[known_aminos[:, 0], known_aminos[:, 1], -5:-2]
    known_pred = pred_coords[known_aminos[:, 0], known_aminos[:, 1], -5:-2]

    # Check input shapes
    if known_true.shape != known_pred.shape:
        raise ValueError("True and predicted coordinates must have the same shape")

    # Calculate pairwise distances
    distances = np.sqrt(np.sum((known_true - known_pred)**2, axis=1))

    # Calculate points within each threshold
    threshold_counts = []
    for threshold in distance_thresholds:
        # Count points within the current threshold
        points_within_threshold = np.sum(distances <= threshold)
        threshold_counts.append(points_within_threshold)
    # Calculate GDT_TS score (average percentage of points within thresholds)
    total_points = known_true.shape[0]
    gdt_ts_percentages = [count / total_points * 100 for count in threshold_counts]
    gdt_ts_score = np.mean(gdt_ts_percentages)

    return gdt_ts_score, dict(zip(distance_thresholds, threshold_counts))



def defined_accuracy(true, pred):
    """
    Finds our model's accuracy in predicting whether atom has defined position
    :param given:
    :param tar:
    :return:
    """
    true_coords = np.array(true)
    pred_coords = np.array(pred)

    # get the lists of coordinates only for atoms that belong in the amino
    known_aminos = np.argwhere(true_coords[:, :, -1] == 1)

    known_true = true_coords[known_aminos[:, 0], known_aminos[:, 1], -5:]
    known_pred = pred_coords[known_aminos[:, 0], known_aminos[:, 1], -5:]
    # for i in known_aminos:
    #     print(f'known: {true_coords[i[0], i[1], -5:]} {true_coords[i[0], i[1], 1]}')
    #     print(f'pred : {pred_coords[i[0], i[1], -5:]} {pred_coords[i[0], i[1], 1]}')

    # for j, k in zip(known_true, known_pred):
    #     print(f'true: {j}')
    #     print(f'pred: {k}')

    condition1 = (known_true[:, -2] == 1)  # Condition: value in array1 == 1
    condition2 = (known_pred[:, -2] > 0.5)  # Condition: value in array2 > 0.5

    combined_condition = condition1 & condition2
    yes_pos = np.sum(combined_condition)
    total_y = np.sum(condition1)

    condition1 = (known_true[:, -2] == 0)  # Condition: value in array1 == 1
    condition2 = (known_pred[:, -2] < 0.5)  # Condition: value in array2 > 0.5

    combined_condition = condition1 & condition2
    no_neg = np.sum(combined_condition)
    total_n = np.sum(condition1)


    correct_pos = yes_pos / total_y
    correct_neg = no_neg / total_n
    # print(f'Correct Pos: {yes_pos / total_y}')
    # print(f'Correct Neg: {no_neg / total_n}')

    return correct_pos, correct_neg





def predict_codes(seq, tar, model):
    """
    Manually return prediction from directories of processed protein files

    Takes a directory of in and out files, returns predictions of those files

    :param in_dir:
    :param out_dir:
    :return: true values, predictions
    """
    model.eval()

    print(f'num proteins in sample: {seq.shape[0] // 1000}')

    criterion = RMSDLoss()

    esm_model, esm_alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_batch_converter = esm_alphabet.get_batch_converter()
    esm_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    esm_model = esm_model.to(device)

    pdb_dir = "UNUSED_WILL_REMOVE"
    data_size = seq.shape[0]
    dataset = ProteinStructureDataset(pdb_dir, esm_model, esm_batch_converter, seq, tar, device,
                                      num_training=data_size)
    batch_size = 10
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn_two)


    num_batches = 10
    tm_scores = np.zeros((batch_size * num_batches))
    losses = np.zeros((batch_size * num_batches))
    gdts = np.zeros((batch_size * num_batches))
    classifications = np.zeros((batch_size * num_batches, 2))
    i = 0
    # for each batch...
    for batch_idx, (seq_emb, coords_true) in enumerate(dataloader):
        coords_true = coords_true.to(device)
        seq_emb = seq_emb.to(device)
        # predict the coordinates from the sequence embedding
        s = time.time_ns()
        coords_pred = model(seq_emb)
        e = time.time_ns()
        # print(f'took: {e - s} nanoseconds')

        # bring true and predicted coordinates to cpu
        ct_cpu = coords_true.to('cpu')
        cp_cpu = coords_pred.to('cpu')

        # Get the size of the batch
        B = ct_cpu.shape[0]

        # Take coordinates from (B,N,2430) into (N*B,27,90)
        unstacked_true = unstack(ct_cpu)
        unstacked_pred = unstack(cp_cpu)

        # Take coordinates from (N*B,27,90) into (B,N,27,90)
        unstacked_true = unstacked_true.reshape(B, -1, 27, 90)
        unstacked_pred = unstacked_pred.reshape(B, -1, 27, 90)

        # (5, 1000, 27, 90)
        # Batch size, sequence length, number of atoms, number of dimensions in each atom (only need last 3)

        # Calculate tm score for each protein in batch
        for j in range(B):
          tm_scores[i * batch_size + j] = calculate_tm_score(unstacked_true[j], unstacked_pred[j])
          # Calculate GDT TS
          gdt_ts_score, threshold_details = calculate_gdt_ts(unstacked_true[j], unstacked_pred[j])
          gdts[i * batch_size + j] = gdt_ts_score

          classifications[i * batch_size + j, :] = defined_accuracy(unstacked_true[j], unstacked_pred[j])
          # print(f"GDT_TS Score: {gdt_ts_score:.2f}")

        losses[i * batch_size:(i + 1) * batch_size] = criterion(coords_pred, coords_true).to('cpu')
        # print(f'losses: {losses}')


        # increase our batch number, break if we've searched all batches!
        i += 1
        print(f'computed batch {i}')
        if i >= num_batches:
          break

    # tm-score for this input
    # print(f"TM-Scores: {tm_scores}")


    return tm_scores, losses, gdts, classifications




def print_prediction(unstacked_true, unstacked_pred):
  """
  Display the result of our single prediction
  """
  for i in range(5):
          for t, p in zip(unstacked_true[i, :, -5:], unstacked_pred[i, :, -5:]):
              print(f'true: {t}')
              print(f'pred: {p}')
              print()
          print()


def plot_acc_loss(node_name):
    train_acc = np.load(f'PDBs/accuracies/train_acc/{node_name}.npy', mmap_mode='r', allow_pickle=True)
    valid_acc = np.load(f'PDBs/accuracies/valid_acc/{node_name}.npy', mmap_mode='r', allow_pickle=True)

    train_loss = np.load(f'PDBs/accuracies/train_loss/{node_name}.npy', mmap_mode='r', allow_pickle=True)
    valid_loss = np.load(f'PDBs/accuracies/valid_loss/{node_name}.npy', mmap_mode='r', allow_pickle=True)


    train_class = np.load(f'PDBs/accuracies/train_class/{node_name}.npy', mmap_mode='r', allow_pickle=True)[0, :]
    valid_class = np.load(f'PDBs/accuracies/valid_class/{node_name}.npy', mmap_mode='r', allow_pickle=True)[0, :]

    print(f'train class shape: {train_class.shape}')

    train_acc = train_acc.squeeze()
    valid_acc = valid_acc.squeeze()

    train_loss = train_loss.squeeze()
    valid_loss = valid_loss.squeeze()

    train_class = train_class.squeeze()
    valid_class = valid_class.squeeze()

    # t = np.arange(0, 2000, 10)
    # even though step size is 10, we only saved every 100 proteins, so each save represents 100 proteins.
    # 100 * 10 = 1000
    t = np.arange(0, 2000 * 100, 10 * 100)

    plt.figure(figsize=(10, 10))

    plt.plot(t, train_acc, label=f'training')
    plt.plot(t, valid_acc, label=f'validation')

    plt.title(f'Training accuracy of model {node_name[-3:-1]}')
    plt.xlabel(f'Samples trained on')
    plt.ylabel(f'TM-score')
    plt.legend()

    plt.savefig(f'Results/2-{node_name}-Accuracy.png')
    # plt.figure(figsize=(8, 8))
    fig, ax1 = plt.subplots(figsize=(8, 8))

    color = 'tab:red'
    color2 = 'lightcoral'
    ax1.set_xlabel(f'Samples trained on')
    ax1.set_ylabel(f'RMSD Loss', color=color)
    ax1.plot(t, train_loss, label=f'Loss training', color=color2)
    ax1.plot(t, valid_loss, label=f'Loss validation', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'blue'
    color2 = 'lightblue'
    ax2.set_ylabel(f'Classification Accuracy', color=color)
    ax2.plot(t, train_class, label=f'Classification training', color=color2)
    ax2.plot(t, valid_class, label=f'Classification validation', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()

    # plt.plot(t, train_class, label=f'training classification')
    # plt.plot(t, valid_class, label=f'validation classification')

    plt.title(f'Training Loss of model {node_name[-3:-1]}')
    plt.legend()
    # plt.xlabel(f'Samples trained on')
    # plt.ylabel(f'RMSD Loss')
    # plt.legend()
    plt.savefig(f'Results/2-{node_name}-Loss.png')




if __name__ == '__main__':
    print(f'sys argv: {sys.argv}')
    if len(sys.argv) > 1:
        node_name = sys.argv[1]

    else:
        node_name = 'DH01A'

    # ---------------------- Logging framework ----------------------
    os.makedirs(f'Logs/{node_name}', exist_ok=True)
    # 10MB handlers
    file_handler = logging.handlers.RotatingFileHandler(f'Logs/{node_name}/Full_Log.log', maxBytes=10000000,
                                                        backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    # Starts each call as a new log!
    file_handler.doRollover()

    master_handler = logging.FileHandler(f'Logs/{node_name}/ERRORS.log', mode='w')
    master_handler.setLevel(logging.WARNING)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, master_handler],
                        format='%(levelname)-8s: %(asctime)-22s %(module)-20s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S | ')
    logging.warning(f'starting with node: {node_name}')
    logging.info(f'{sys.argv}')

    print(f'Started with following system variables:')
    print(f'{sys.argv}')
    print(f'node_name: {node_name}')

    # ---------------------- End Logging Framework ----------------------


    # how large to step for each accuracy
    step_size = 10
    max = 2000
    i = 0

    train_acc = np.memmap(f'PDBs/accuracies/train_acc/{node_name}', dtype='float32', mode='w+', shape=(1, max // step_size))
    valid_acc = np.memmap(f'PDBs/accuracies/valid_acc/{node_name}', dtype='float32', mode='w+', shape=(1, max // step_size))

    train_loss = np.memmap(f'PDBs/accuracies/train_loss/{node_name}', dtype='float32', mode='w+', shape=(1, max // step_size))
    valid_loss = np.memmap(f'PDBs/accuracies/valid_loss/{node_name}', dtype='float32', mode='w+', shape=(1, max // step_size))

    train_class = np.memmap(f'PDBs/accuracies/train_class/{node_name}', dtype='float32', mode='w+', shape=(2, max // step_size))
    valid_class = np.memmap(f'PDBs/accuracies/valid_class/{node_name}', dtype='float32', mode='w+', shape=(2, max // step_size))
    # train_acc = np.zeros((9, max // step_size))
    # valid_acc = np.zeros((9, max // step_size))

    seq_train = np.load('PDBs/big_data/tests/in-train.npy', mmap_mode='r', allow_pickle=True)
    tar_train = np.load('PDBs/big_data/tests/out-train.npy', mmap_mode='r', allow_pickle=True)

    # seq_train = np.load('PDBs/big_data/tests/train_manual-in.npy', mmap_mode='r', allow_pickle=True)
    # tar_train = np.load('PDBs/big_data/tests/train_manual-out.npy', mmap_mode='r', allow_pickle=True)
    # only want the first 100 samples
    # recall samples are stacked on top of one another
    seq_train = seq_train[:1000 * 100]
    tar_train = tar_train[:1000 * 100, :]

    seq_valid = np.load('PDBs/big_data/tests/in-valid.npy', mmap_mode='r', allow_pickle=True)
    tar_valid = np.load('PDBs/big_data/tests/out-valid.npy', mmap_mode='r', allow_pickle=True)

    seq_valid = seq_valid[:1000 * 100]
    tar_valid = tar_valid[:1000 * 100, :]

    # -------------------- Single prediction ----------------
    # torch.set_grad_enabled(False)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = open_model(f'models/{node_name}/Save-2000')
    #
    # # calculate accuracy for training and validation
    # tm_scores, losses, gdts_t, class_t = predict_codes(seq_train, tar_train, model)
    # avg_train = np.average(tm_scores)
    # avg_tLoss = np.average(losses)
    # avg_tGDT = np.average(gdts_t)
    # avg_tClass = np.average(class_t, axis=0)
    #
    # tm_scores, losses, gdts_v, class_v = predict_codes(seq_valid, tar_valid, model)
    # avg_valid = np.average(tm_scores)
    # avg_vLoss = np.average(losses)
    # avg_vGDT = np.average(gdts_v)
    # avg_vClass = np.average(class_v, axis=0)
    #
    #
    # # calculate loss for training and validation
    #
    # # print(f'all train score: {tm_scores}')
    # print(f'avg train score: {avg_train}')
    # print(f'avg train gdt: {avg_tGDT}')
    # print(f'train max gdt: {np.max(gdts_t)}')
    # print(f'train classification: {avg_tClass}')
    # # print(f'training gdts: {gdts_t}')
    # # print(f'training classes: {class_t}')
    #
    # print()
    #
    # print(f'avg valid score: {avg_valid}')
    # print(f'avg valid gdt: {avg_vGDT}')
    # print(f'valid max gdt: {np.max(gdts_v)}')
    # # print(f'validation gdts: {gdts_v}')
    # print(f'valid classification: {avg_vClass}')
    # # print(f'validation classes: {class_v}')
    #
    #
    # pause




    # r = RMSDLoss()
    # print(f'train acc: {train_acc.shape}')
    # for n in range(0, max, step_size):
    #     print(f'n: {n}')
    #     logging.info(f'- n {n} -')
    #     torch.set_grad_enabled(False)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = open_model(f'models/{node_name}/Save-{i}')
    #
    #     # calculate accuracy for training and validation
    #     tm_scores, losses, gdts_t, class_t = predict_codes(seq_train, tar_train, model)
    #     avg_train = np.average(tm_scores)
    #     avg_tLoss = np.average(losses)
    #     # avg_tGDT = np.average(gdts_t)
    #     avg_tClass = np.average(class_t, axis=0)
    #
    #     tm_scores, losses, gdts_v, class_v = predict_codes(seq_valid, tar_valid, model)
    #     avg_valid = np.average(tm_scores)
    #     avg_vLoss = np.average(losses)
    #     # avg_vGDT = np.average(gdts_v)
    #     avg_vClass = np.average(class_v, axis=0)
    #
    #     # calculate loss for training and validation
    #
    #
    #     # print(f'avg train score: {avg_train}')
    #     print(f'avg train score: {avg_train}')
    #     print(f'avg valid score: {avg_valid}')
    #
    #     train_acc[0, i] = avg_train
    #     valid_acc[0, i] = avg_valid
    #
    #     train_loss[0, i] = avg_tLoss
    #     valid_loss[0, i] = avg_vLoss
    #
    #     train_class[:, i] = avg_tClass
    #     valid_class[:, i] = avg_vClass
    #
    #     logging.info(f'Added all nodes for iter {n}')
    #     print()
    #     i += 1
    #
    # # print(f'train accuracies: {train_acc}')
    # # print(f'valid accuracies: {valid_acc}')
    # np.save(f'PDBs/accuracies/train_acc/{node_name}', allow_pickle=True, arr=train_acc)
    # np.save(f'PDBs/accuracies/valid_acc/{node_name}', allow_pickle=True, arr=valid_acc)
    #
    # np.save(f'PDBs/accuracies/train_loss/{node_name}', allow_pickle=True, arr=train_loss)
    # np.save(f'PDBs/accuracies/valid_loss/{node_name}', allow_pickle=True, arr=valid_loss)
    #
    # np.save(f'PDBs/accuracies/train_class/{node_name}', allow_pickle=True, arr=train_class)
    # np.save(f'PDBs/accuracies/valid_class/{node_name}', allow_pickle=True, arr=valid_class)


    # ----------------------------- Plot accuracies and losses
    # train_acc = np.load('PDBs/accuracies/train_acc/DH01A.npy', mmap_mode='r', allow_pickle=True)
    # valid_acc = np.load('PDBs/accuracies/valid_acc/DH01A.npy', mmap_mode='r', allow_pickle=True)
    #
    # train_loss = np.load('PDBs/accuracies/train_loss/DH01A.npy', mmap_mode='r', allow_pickle=True)
    # valid_loss = np.load('PDBs/accuracies/valid_loss/DH01A.npy', mmap_mode='r', allow_pickle=True)
    #
    #
    # print(f'train acc: {train_acc}')
    # print(f'valid acc: {valid_acc}')
    #
    # print(f'train loss: {train_loss}')
    # print(f'valid loss: {valid_loss}')
    # for i in range(9):
    #     i += 1
    #     name = f'DH0' + str(i) + 'A'
    #     plot_acc_loss(name)
    plot_acc_loss('DH08A')
    # plot_acc_loss('DH01A')
    # plt.show()

