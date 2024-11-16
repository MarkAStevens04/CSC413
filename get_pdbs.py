from Bio import PDB
from Bio.PDB import PDBList
from Bio.PDB import Selection
from Bio.PDB import Polypeptide
from Bio.PDB.Polypeptide import PPBuilder
import glob2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import amino_expert as aa

# run "pip install -r requirements.txt"

PID_DIR = "PDBs/protein-ids.txt"
mmCIF_DIR = "PDBs/all"

parser = PDB.MMCIFParser(QUIET=True)

def download_list(max_download=10):
    """
    Download PDB files from rcsb. Uses PID_DIR to get list of proteins to download.
    Returns the list of all PDBs being used

    :param max_download: Maximum number of downloads.
    :return: list of downloaded proteins
    """
    # each entry is roughly 1MB (Downloading 1,000 PDB would be roughly 1GB)
    pdb_list = PDBList()
    # get all 4 letter codes of desired proteins
    names = open(PID_DIR, "r+").readline().split(',')

    using = []
    for i in range(min(max_download, len(names))):

        pdb_filename = pdb_list.retrieve_pdb_file(names[i], pdir=mmCIF_DIR, file_format="mmCif")
        using.append(names[i])
        # print(f'{pdb_filename}')
    # print(f'curr names: {names}')
    return using


def calc_dist_mat(p_struct):
    """
    Calculates the distance matrix for the specified protein
    :param Protein:
    :return:
    """
    # Helpful for parsing structure
    # https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ

    expert = aa.AA_Expert()

    ca_atoms = []
    for model in p_struct:
        for chain in model:
            # A=atom, R=residue, C=chain, M=model, S=structure.
            residues = Selection.unfold_entities(chain, "R")
            for r, residue in enumerate(chain):

                # here's how to check it's an amino acid residue (standard amino acid)
                print(f'is residue? {Polypeptide.is_aa(residue)} {residue}')
                if Polypeptide.is_aa(residue):
                    expert.train_single_amino(residue)

                if 'CA' in residue:
                    ca_atoms.append(residue['CA'].coord)
        print(f'next chain: {len(ca_atoms)}')

    ppb = PPBuilder()
    for pp in ppb.build_peptides(p_struct):
        print(pp.get_sequence())
        print(f'len: {len(pp.get_sequence())}')

    print(expert)


    # model = p_struct[0]
    # chain = Selection.unfold_entities(model, "C")[0]
    # residues = Selection.unfold_entities(chain, "R")
    # print(f'residues: {residues}')
    # residue = chain[(' ', 135,' ')]
    #
    # print(f'residue: {residue}')

    # for r in range(120, 140):
    #     print(f'--------------- {r}')
    #     residue = chain[r]
    #
    #     print(f'residue: {residue}')
    #     print(f'')

    # residue = chain[100]
    # print(f'residue: {residue}')







    num_atoms = len(ca_atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(ca_atoms[i] - ca_atoms[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix

def get_structs(names, display_first=True):
    """
    Gets the structures from the names list.
    Should be stored in the mmCIF_DIR
    :param names:
    :return:
    """

    # gathers the PDB file for each pdb name in names
    i = 0
    print(f'name: {names}')
    names = [mmCIF_DIR + "/" + name + ".cif" for name in names]
    files = [open(name) for name in names]

    dist_mats = []
    for pid_file in files:
        p_struct = parser.get_structure(pid_file, pid_file)
        dist_mat = calc_dist_mat(p_struct)
        dist_mats.append(dist_mat)

    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_mats[0], cmap="viridis", square=True, cbar_kws={'label': 'Distance (A)'})
    plt.title(f'Contact Map / Distance matrix for {files[0].name}')





if __name__ == '__main__':
    names = download_list()
    # names = ["cat","dog"]
    dist_mat = get_structs(names[0:1])
    plt.show()




