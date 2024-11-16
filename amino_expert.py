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
import os

import get_pdbs as gp

RES_DIR = "PDBs/Residues"





class AA_Expert():
    """
    Class for the amino acid expert.
    Associated Methods for training and using the expert.
    """
    def __init__(self):
        self.aminos = {}
        self.vocab = set()

    def use_ideal_aminos(self):
        """
        Method for training our Amino Acid Expert on ideal amino acids.
        Downloaded from wwPDB
        :return:
        """
        all_files = glob.glob(RES_DIR + '/*.cif')
        for file in all_files: # For every file in our Amino Acid folder
            mmcif_dict = MMCIF2Dict(file) # Turn MMCIF into dictionary
            print(mmcif_dict)
            single_letter = mmcif_dict['_chem_comp.one_letter_code'][0]
            print(f'single letter: {single_letter}')
            amino_atom_list = mmcif_dict['_chem_comp_atom.type_symbol']
            amino_backbone_flags = mmcif_dict['_chem_comp_atom.pdbx_backbone_atom_flag']
            amino_x  =mmcif_dict['_chem_comp_atom.pdbx_model_Cartn_x_ideal']
            amino_y = mmcif_dict['_chem_comp_atom.pdbx_model_Cartn_y_ideal']
            amino_z = mmcif_dict['_chem_comp_atom.pdbx_model_Cartn_z_ideal']


            # Throw error if list sizes are not correct!
            if not (len(amino_x) == len(amino_y) == len(amino_z) == len(amino_atom_list) == len(amino_backbone_flags)):
                print(f'ERROR! Length of input peptide sequence is incorrect.')
                print(f'file: {file}')
                print(f'dict: {mmcif_dict}')
                print(f'len amino_x: {len(amino_x)}')
                print(f'len amino_y: {len(amino_y)}')
                print(f'len amino_z: {len(amino_z)}')
                print(f'len amino_atom_list: {len(amino_atom_list)}')
                print(f'len amino_atom_list: {len(amino_atom_list)}')



            for a, f in zip(amino_atom_list, amino_backbone_flags):
                if f == 'Y':
                    self.vocab.add(a + 'B')
                else:
                    self.vocab.add(a)

            self.aminos[single_letter] = (amino_atom_list, amino_backbone_flags, amino_x, amino_y, amino_z)


        # Encode atom names into integer list
        print(f'built vocab! {self.vocab}')
        possible_atoms = sorted(list(set(self.vocab)))

        stoi = { ch:i for i,ch in enumerate(possible_atoms)}
        itos = { i:ch for i,ch in enumerate(possible_atoms)}

        print(f'stoi: {stoi}')
        print(f'itos: {itos}')

        self.encode = lambda s: [stoi[c] for c in s]
        self.decode = lambda l: ''.join([itos[i] for i in l])

        print(f"encodings: {self.encode(['CB'])}")
        print(f"more enco:  {self.encode(['C'])}")


        for amino_abbrev in self.aminos:
            # print(f'amino dict: {self.aminos[amino_abbrev]}')
            amino_atom_list = self.aminos[amino_abbrev][0]
            amino_backbone_flags = self.aminos[amino_abbrev][1]
            amino_x = self.aminos[amino_abbrev][2]
            amino_y = self.aminos[amino_abbrev][3]
            amino_z = self.aminos[amino_abbrev][4]

            amino_dict = np.random.random((1, 4))
            print(f'starting amino dict: {amino_dict}')


            # Possibly change this in the future to only have NCC contain backbone flag!
            for a, f, x, y, z in zip(amino_atom_list, amino_backbone_flags, amino_x, amino_y, amino_z):

                if f == 'Y':
                    atom_name = a + 'B'
                else:
                    atom_name = a

                entry = [[self.encode([atom_name])[0], float(x), float(y), float(z)]]
                print(f'entry: {entry}')
                amino_dict = np.append(amino_dict, entry, axis=0)

            print(f'full amino dict: {amino_dict}')
            amino_dict = np.delete(amino_dict, 0, axis=0)
            print(f'full amino dict: {amino_dict}')


        # '_chem_comp.one_letter_code': ['A']
        # '_chem_comp_atom.atom_id': ['N', 'CA', 'C', 'O', 'CB', 'OXT', 'H', 'H2', 'HA', 'HB1', 'HB2', 'HB3', 'HXT']
        # '_chem_comp_atom.type_symbol': ['N', 'C', 'C', 'O', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H', 'H'],
        # '_chem_comp_atom.pdbx_backbone_atom_flag': ['Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'Y']
        # '_chem_comp_atom.pdbx_model_Cartn_x_ideal': ['-0.966', '0.257', '-0.094', '-1.056', '1.204', '0.661', '-1.383', '-0.676', '0.746', '1.459', '0.715', '2.113', '0.435']





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
    """
    Downloads all amino acids!
    Source: https://www.rcsb.org/docs/programmatic-access/file-download-services
    Example link: https://files.rcsb.org/ligands/download/HEM.cif
    :return:
    """
    amino_list = ["Ala", "Arg", "Asp", "Asn", "Cys", "Glu", "Gln", "Gly", "His", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"]
    for res in amino_list:
        url = "https://files.rcsb.org/ligands/download/" + res.upper() + ".cif"
        wget.download(url, out=RES_DIR)


parser = PDB.MMCIFParser(QUIET=True)


if __name__ == "__main__":

    # download_all_aminos()
    e = AA_Expert()
    e.use_ideal_aminos()



    # names = gp.download_list()
    # dist_mat = gp.get_structs(names[0:1])
    # plt.show()