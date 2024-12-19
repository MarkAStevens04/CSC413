from Bio import SeqIO
from Bio import PDB
from Bio.PDB import PDBList
from Bio.PDB import Selection
from Bio.PDB import Structure
from Bio.PDB import Polypeptide
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import glob2
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import logging
import logging.handlers
from Bio.SeqUtils import seq1
import re
from Bio.PDB.mmcifio import MMCIFIO
from logging_setup import setup_logger, get_logger, change_log_code

# setup_logger()
# logger = get_logger()
# logger.setLevel(logging.DEBUG)


import amino_expert as aa

# run "pip install -r requirements.txt"

PID_DIR = "PDBs/protein-ids.txt"
mmCIF_DIR = "PDBs/all"

parser = PDB.MMCIFParser(QUIET=True)

def download_list(max_download=30):
    """
    Download PDB files from rcsb. Uses PID_DIR to get list of proteins to download.
    Returns the list of all PDBs being used

    :param max_download: Maximum number of downloads.
    :return: list of downloaded proteins
    """
    # each entry is roughly 1MB (Downloading 1,000 PDB would be roughly 1GB)
    pdb_list = PDBList(server="https://files.wwpdb.org/")
    # pdb_list = PDBList(server="rsync://rsync.rcsb.org")

    # get all 4 letter codes of desired proteins
    names = open(PID_DIR, "r+").readline().replace('\n', '').split(',')

    using = []
    # for i in range(min(max_download, len(names))):
        # pdb_filename = pdb_list.retrieve_pdb_file(names[i], pdir=mmCIF_DIR, file_format="mmCif")
        # pdb_files = pdb_list.download_pdb_files(pdb_codes=names[:max_download], pdir=mmCIF_DIR, file_format="mmCif")
        # pdb_filename = pdb_list.retrieve_assembly_file(names[i], assembly_num = '1', pdir=mmCIF_DIR, file_format="mmCif")
        # pdb_filename = pdb_list.download_all_assemblies(names[i], pdir=mmCIF_DIR, file_format="mmCif")
        # using.append(names[i])

        # base_url = "https://files.rcsb.org/download/"
        # url = f"{base_url}{names[i].lower()}-assembly1.cif"
        #
        # # File path to save the downloaded file
        # file_path = f"{mmCIF_DIR}/{names[i].lower()}_assembly1.cif"
        #
        # # Download the file
        # response = requests.get(url)
        # if response.status_code == 200:
        #     with open(file_path, "wb") as file:
        #         file.write(response.content)
        #     print(f"Downloaded biological assembly to {file_path}")
        #     return file_path
        # else:
        #     raise Exception(f"Failed to download file. HTTP Status: {response.status_code}")
        #
        # using.append(names[i])

    pdb_files = pdb_list.download_pdb_files(pdb_codes=names[:max_download], pdir=mmCIF_DIR, file_format="mmCif")
    using.extend(names[:max_download])

    return using


def calc_dist_mat(p_struct):
    """
    Calculates the distance matrix for the specified protein
    :param Protein:
    :return:
    """
    # Helpful for parsing structure
    # https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ

    ca_atoms = []
    for model in p_struct:
        for chain in model:
            # A=atom, R=residue, C=chain, M=model, S=structure.
            # residues = Selection.unfold_entities(chain, "R")
            for r, residue in enumerate(chain):

                # here's how to check it's an amino acid residue (standard amino acid)
                # print(f'is residue? {Polypeptide.is_aa(residue)} {residue}')

                if 'CA' in residue:
                    ca_atoms.append(residue['CA'].coord)
        # print(f'next chain: {len(ca_atoms)}')



    # Creates the distance matrix
    num_atoms = len(ca_atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = np.linalg.norm(ca_atoms[i] - ca_atoms[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance

    return distance_matrix



def sequence_construction(p_struct: PDB.Structure.Structure, seq):
    """ Align protein sequences of known vs unknown residue positions, extract atom positions

    Aligns seq with the sequence found in p_struct.
    In the process, extracts atomic coordinates.
    Difficult because p_struct may have multiple (overlapping) chains


    Outputs the original sequence alongside the position sequence.
    Used to determine which amino acids we know the position of, and which we do not.
    = Future optimization: =
     Instead of 2nd string being list of letters and dashes, just have it be 1s and 0s corresponding to
     whether we know the position or not.
    :param p_struct:
    :param seq:
    :return:
    """
    ppb = PPBuilder()
    print(f'seq: {seq}')
    output_str = ''
    prev_idx = 0

    atom_list = []

    for pp in ppb.build_peptides(p_struct):
        start_idx = pp[0].id[1] - 1

        # cut off extra letters!
        # Happens when proteins are defined twice.
        # This is because we only want to train our model on a single sequence at a time.
        # We do this so it can hopefully learn to recognize when multiple proteins might be present.
        remove = max(prev_idx - start_idx, 0)

        output_str += '-' * (start_idx - prev_idx)
        output_str += str(pp.get_sequence()[remove:])


        # populate our atom list with the new residues!
        res_list = pp[remove:]
        for res in res_list:
            single_res = [res.id[1], seq1(res.get_resname())]
            for atom in res:  # Loop through atoms
                atom_name = atom.get_name()
                coords = atom.get_coord()  # Get atomic coordinates (x, y, z)
                # a = [atom.get_id(), atom.get_coord(), atom.get_parent()]
                single_res.append((atom_name, coords))
            atom_list.append(single_res)

        # keep the highest from prev_idx and end of this string.
        # We already added elements up to prev_idx, we don't want to add them again
        # in a later string!
        prev_idx = max(pp[-1].id[1], prev_idx)

    # Repeat again at the end in case our protein ends with an unknown
    start_idx = len(seq)
    output_str += '-' * (start_idx - prev_idx)


    print(f'out: {output_str}')
    return seq, output_str, atom_list


def atom_interpreter(p_struct: PDB.Structure.Structure, pos_seq):
    """ NOT USED

    NOT USED
    Returns the positions of all known atoms!
    Takes pos_seq so we know which residues to pull atomic coords from.
    :param p_struct:
    :return:
    """
    # # print(f'all residues: {p_struct.get_residues()}')
    # atoms = []
    # for r in p_struct.get_residues():
    #     # print(f'residue: {r}')
    #     res_position = r.id[1]
    #     # print(f'res ---- {r}')
    #     # for n in r.get_iterator():
    #     #     print(n)
    #
    #     if r.is_disordered():
    #         logger.warning(f'Non-standard residue! {p_struct}, {r}, {r.id}')
    #     else:
    #         a_sublist = [res_position, r.get_resname()]
    #         for atom in r.get_unpacked_list():
    #             # print(f'atom name: {atom}')
    #             # print(f'atom: {atom.get_id()}, {atom.get_coord()}, {residue.id, }')
    #             a = [atom.get_id(), atom.get_coord()]
    #             a_sublist.append(a)
    #         atoms.append(a_sublist)

    atoms = []
    for model in p_struct:  # Loop through models (may be multiple)
        for chain in model:  # Loop through chains

            for residue in chain:  # Loop through residues
                # print(f'res ----- {residue}')
                single_res = [residue.id[1], seq1(residue.get_resname())]
                for atom in residue:  # Loop through atoms
                    atom_name = atom.get_name()
                    coords = atom.get_coord()  # Get atomic coordinates (x, y, z)
                    a = [atom.get_id(), atom.get_coord(), atom.get_parent()]
                    # print(f'a: {a}')
                    single_res.append((atom_name, coords))
                atoms.append(single_res)


    return atoms




def polypeptide_interpreter(p_struct: PDB.Structure.Structure, seq):
    """
    Interprets the peptide segments in our protein structure
    :return:
    """
    print(f'polypeptide builder...')
    print(p_struct)
    for model in p_struct:
        for chain in model:
            residues = Selection.unfold_entities(chain, "R")
            print(residues)
            for residue in chain:
                print(residue.id)



    print(f'starting...')
    ppb = PPBuilder()

    for pp in ppb.build_peptides(p_struct):
        # gets the id of the first amino acid in the string
        print(pp[0])
        start = pp[0].id[1]
        end = pp[-1].id[1]
        print(f'original: {seq[start-1:end]}')
        print(f'modified: {pp.get_sequence()}')


        print(f'start: {start}, end: {end}')
        print(pp.get_sequence())
        print(pp)
        print(f'len: {len(pp.get_sequence())}')

    print(f'seq: {seq}')
    output_str = 'ppt: '
    prev = 0
    for pp in ppb.build_peptides(p_struct):
        start = pp[0].id[1] - 1
        output_str += '-' * (start - prev)
        output_str += pp.get_sequence()
        prev = pp[-1].id[1]
    start = len(seq)
    output_str += '-' * (start - prev)
    print(output_str)






def get_structs(names):
    """ Get sequences and atom positions from protein code.

    Goal: Extract full protein sequence.
    Requires: Extracting only proteins (not DNA or other ligands) (surprisingly difficult)
    Requires: Extracting full protein sequence (not just atoms with known positions) (surprisingly difficult!!)
    Requires: Cross-referencing with cif parser to obtain final sequence
    Requires: Aligning sequences of aminos with known vs unknown positions

    First obtains the "reference" sequence of the protein by looking at mmCif file.
    Then extracts the "known position" sequence of the protein by comparing the peptide list with the reference sequence.
    Performs a pseudo sequence alignment using the position of each amino acid.


    = Possible extensions =
    - allow for input of different annotations. "artifact", mutation, etc.
    - multi-sequence allignment? Perform an allignment to see if these structure motifs already exist in wild
    - ADD OUTPUT OF SEQUENCE ANNOTATIONS?? Prediction on where DNA would bind, stuff like that?
            - Info is found on sequence annotations
            - *** Predict under what conditions protein will crystallize best?? (_exptl_crystal_grow.pdbx_details, _exptl_crystal_grow.method, _exptl_crystal_grow.temp)
                - From the key dict: _exptl_crystal_grow.pdbx_details
            - *** Predict what type of method will be best? (_exptl.method) '_exptl.method'
    - Maybe have our output also include molecule details like heteroatom, binding, etc.

    :param names: List of 4-letter protein codes. Should be stores in mmCIF_DIR/CODE.cif
    :return: (4-Letter Code, Reference Sequence, Known Position Sequence, List of aminos with atom positions)
    """
    logger = get_logger()

    names = [mmCIF_DIR + "/" + name.lower() + ".cif" for name in names if name != '']
    files = [open(name) for name in names]

    # Goal: Extract full protein sequence.
    # Requires: Extracting only proteins (not DNA or other ligands) (surprisingly difficult)
    # Requires: Extracting full protein sequence (not just atoms with known positions) (surprisingly difficult!!)
    # Requires: Cross-referencing with cif parser to obtain final sequence
    # Requires: Aligning sequences of aminos with known vs unknown positions

    # to investigate:
    # _struct_ref.pdbx_seq_one_letter_code
    # _struct_ref_seq.pdbx_strand_id
    # _struct_ref_seq.seq_align_beg
    # _struct_ref_seq.db_align_beg
    # _struct_ref_seq.pdbx_auth_seq_align_beg
    # _struct_ref_seq.ref_id


    # -------------------------- SOLUTION!!!!!! ------------------------------
    # _entity_poly.type:
    # _entity_poly.pdbx_seq_one_letter_code: Maybe don't need to use this one?
    # _entity_poly.pdbx_strand_id: ['A,C', 'B,E', 'D,F'] (D,F)
    # _pdbx_poly_seq_scheme.ndb_seq_num
    # _pdbx_poly_seq_scheme.seq_id
    # _pdbx_poly_seq_scheme.entity_id
    #


    sequences = []
    # for every mmCif file and associated directory...
    for pid_file, name in zip(files, names):
        change_log_code(name[-8:-4])
        # extract the mmCif dictionary to extract specific attributes
        pdb_info = MMCIF2Dict(name)

        # print(f'()()()()()()()()()()()()()()()()()()()()()()()()()()()() {name}')
        print()
        print(f'{name}')

        ptein_peps = []
        ref_seq = ''
        # for each peptide in _entity_poly.entity_id ...
        # goal is to get the ACTUAL start position. Cannot access this info with regular methods
        # because the residues are not populated in any objects because biopython does not recognize this sequence.
        for pep_id in pdb_info['_entity_poly.entity_id']:
            found = False

            # id is that noted in the mmCif file - 1 (0-indexed vs 1-indexed)
            id = int(pep_id) - 1
            type = pdb_info['_entity_poly.type'][id]
            if type == 'polypeptide(L)':
                # Only extract actual polypeptides!
                # Cannot use name found from iterating over each chain, it says DNA & Proteins are both polypeptides (true but not helpful)

                for chain_id, start_idx, chain_name in zip(pdb_info['_pdbx_poly_seq_scheme.entity_id'],
                                               pdb_info['_pdbx_poly_seq_scheme.pdb_seq_num'],
                                               pdb_info['_pdbx_poly_seq_scheme.asym_id']):
                    if chain_id == str(id + 1) and not found:
                        # If the chain id hasn't been located yet...

                        # ptein_peps is technically not required, but extremely useful for debugging.
                        # Denotes the start location of each peptide sequence.
                        ptein_peps.append((chain_name, start_idx))
                        found = True

                        # Add dashes if the polypeptide sequence is separated by non-proteins.
                        # Occurs when sequence is put between DNA sequences or other polypeptides.
                        ref_seq += '-' * (int(start_idx) - 1 - len(ref_seq))
                        processed_seq = pdb_info['_entity_poly.pdbx_seq_one_letter_code'][id].replace('\n', '')
                        # print(f'pre-: {processed_seq}')
                        # Surprisingly large effect when removing (XXX) with -.
                        # Roughly 29/100 -> 19/100
                        processed_seq = re.sub(r'\(.*?\)', '-', processed_seq)
                        # print(f'post: {processed_seq}')
                        # processed_seq = processed_seq.replace('(', '').replace(')', '')
                        ref_seq += processed_seq

                        # The offset was to help fix the bug, but I cannot get it to work :(
                        # if int(start_idx) < 0:
                        #     offset = max(-1 * (int(start_idx) - 1), offset)
                        #     print(f'offset changed to: {offset}')

        # print(f'ptein peps: {ptein_peps}')
        # print(f'')
        # Essentially, if the beginning of the sequence is just filler (hence why its index is below 1), we cut it off!
        # Surprisingly large effect on the data kept. Went from 31/100 gone to 19/100 gone
        if len(ptein_peps) >= 1 and len(ptein_peps[0]) >= 2 and int(ptein_peps[0][1]) <= 0:
            # print(f'pre seq: {ref_seq}')
            ref_seq = ref_seq[-1 * int(ptein_peps[0][1]) + 1:]
            # print(f'postseq: {ref_seq}')

        # get the structure of the protein for extracting atom positions and known-position residue sequence
        p_struct = parser.get_structure(pid_file, pid_file)
        ref_seq, pos_seq, atom_list = sequence_construction(p_struct, ref_seq)

        # entry to our list of proteins
        ptein_entry = (name, ref_seq, pos_seq, atom_list)


        # Error checking...
        # Fix these errors!!!!
        # Count the number of error messages we send
        i = 0
        error = False
        # For each letter in the code...
        for s, o in zip(ptein_entry[1], ptein_entry[2]):
            # check if there's a mismatch...
            if s != '-' and o != '-' and s != o:
                if not error:
                    # logger.warning(f'Invalid Protein!!! {name}')
                    logger.warning(f'mismatch: {s}{o} @ {i}')
                    # logger.debug(f'first mismatch: {s}{o} @ {i}')
                error = True

            # check if we do NOT know the position in the original, but DO know the position
            # in the new one
            if s == '-' and o != '-':
                if error:
                    logger.warning(f'Known position sequence not in reference sequence! {name}')
                    logger.warning(f'reference: {s}, known-position: {o}')
                error = True
            i += 1

        # Finally, check the length of the sequences
        if len(ptein_entry[1]) != len(ptein_entry[2]):
            logger.warning(f'Length Error! Reference: {len(ptein_entry[1])}, known-position: {len(ptein_entry[2])} {name}')
            logger.warning(f'Reference: {ptein_entry[1]}')
            logger.warning(f'Known-pos: {ptein_entry[2]}')
            error = True

        if not error:
            sequences.append(ptein_entry)
        else:
            logger.error(f'Invalid Protein!! {name}')
    change_log_code()
    return sequences


def obtain_training():
    """
    Obtains the training data from our PDB list!
    :return:
    """
    names = download_list()
    dist_mat = get_structs(names[0:6])
    plt.show()




def adjust_position(name, atoms, amino_expert):
    """
    Adjusts the position of some atoms in a mmCif file and saves the file!
    :return:
    """
    print(f'adjusting position...')
    n = mmCIF_DIR + "/" + name.lower() + ".cif"
    file = open(n)
    p_struct = parser.get_structure(name, file)
    i = 0
    print(f'amino decode: {amino_expert.decode}')
    for model in p_struct:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    # Example: Adjusting x, y, z coordinates
                    x, y, z = atom.coord  # Get current coordinates
                    atom.coord = [x + 1.0, y - 1.0, z + 2.0]  # Modify coordinates
                    while atoms[i][4] == 0 and i < len(atoms):
                        i += 1
                    print(f'row: {atoms[i]} {amino_expert.decode[atoms[i][0]]}')
                    print(f'atom: {atom}')
                    i += 1

    io = MMCIFIO()
    io.set_structure(p_struct)
    io.save("output.cif")




if __name__ == '__main__':
    # obtain_training()
    # names = download_list()
    # dist_mat = parse_names(names[0:1])
    # plt.show()


    adjust_position('5RW2')

    # Notable proteins:
    # 6L6Y, 6JUX, 6KNM, 6JHD, 6WNX, 6XBJ, 6Z6U, 6PHN
