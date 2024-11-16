from Bio import PDB
from Bio.PDB import PDBList
from Bio.PDB import Selection
from Bio.PDB import Polypeptide
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import Residue
import glob2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import get_pdbs as gp





class AA_Expert():
    """
    Class for the amino acid expert.
    Associated Methods for training and using the expert.
    """
    def __init__(self):
        self.aminos = {}

    def train_single_amino(self, Res: Residue.Residue):
        """
        :param Res: Residue to train our model on
        :return: None
        """
        # print(f'amino res: {Res.get_resname()}')
        name = Res.get_resname()
        if name not in self.aminos:
            self.aminos[name] = [Res]
        else:
            self.aminos[name].append(Res)
        if Res.is_disordered():
            print(f'UH OH! This amino is disordered!')








if __name__ == "__main__":
    names = gp.download_list()
    # names = ["cat","dog"]
    dist_mat = gp.get_structs(names[0:1])
    plt.show()