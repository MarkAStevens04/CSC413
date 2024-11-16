from Bio import PDB
from Bio.PDB import PDBList
from Bio.PDB import Selection
from Bio.PDB import Polypeptide
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import Residue
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import glob2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import urllib.request
import wget

import get_pdbs as gp

RES_DIR = "PDBs/Residues"





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

        Ca = None
        if 'CA' in Res:
            Ca = Res['CA'].coord
        for atom in Res.get_atoms():
            print(f'atom: {atom}')


    def finalize_aminos(self):
        new_positions = dict()
        for amino_list in self.aminos:
            amino = np.r

    def __str__(self):
        msg = f'All self aminos: \n'
        msg = msg + str(self.aminos) + '\n'
        msg += f'Num aminos: {len(self.aminos)} \n'
        return msg



def download_all_aminos():
    amino_list = ["Ala", "Arg", "Asp", "Asn", "Cys", "Glu", "Gln", "Gly", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
    # urllib.request.urlretrieve("http://www.example.com/songs/mp3.mp3", "mp3.mp3")
    for res in amino_list:
        # print(res.upper())
        url = "https://files.rcsb.org/ligands/download/" + res.upper() + ".cif"
        wget.download(url, out=RES_DIR)


parser = PDB.MMCIFParser(QUIET=True)


if __name__ == "__main__":
    # Source: https://www.rcsb.org/docs/programmatic-access/file-download-services
    # https://files.rcsb.org/ligands/download/HEM_ideal.mol2
    # https://files.rcsb.org/ligands/download/HEM.cif


    # download_all_aminos()

    print(f"retrieved")
    name = "PDBs/Residues/ALA.cif"
    file = open(name)

    mmcif_dict = MMCIF2Dict(name)


    # p_struct = parser.get_structure(file, file)

    print(f'Printing')
    print(mmcif_dict)
    print(f'complete')
    # '_chem_comp.one_letter_code': ['A']
    # '_chem_comp_atom.atom_id': ['N', 'CA', 'C', 'O', 'CB', 'OXT', 'H', 'H2', 'HA', 'HB1', 'HB2', 'HB3', 'HXT']
    # '_chem_comp_atom.type_symbol': ['N', 'C', 'C', 'O', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
    # '_chem_comp_atom.pdbx_backbone_atom_flag': ['Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'Y']
    # '_chem_comp_atom.pdbx_model_Cartn_x_ideal': ['-0.966', '0.257', '-0.094', '-1.056', '1.204', '0.661', '-1.383', '-0.676', '0.746', '1.459', '0.715', '2.113', '0.435']








    # names = gp.download_list()
    # # names = ["cat","dog"]
    # dist_mat = gp.get_structs(names[0:1])
    # plt.show()