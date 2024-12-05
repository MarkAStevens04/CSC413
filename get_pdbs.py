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
    # get all 4 letter codes of desired proteins
    names = open(PID_DIR, "r+").readline().split(',')

    using = []
    for i in range(min(max_download, len(names))):
        pdb_filename = pdb_list.retrieve_pdb_file(names[i], pdir=mmCIF_DIR, file_format="mmCif")
        using.append(names[i])

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
    """
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
    output_str = 'ppt: '
    prev = 0
    # for pp in ppb.build_peptides(p_struct):
    #     print(f'peptide pp: {pp.get_sequence()}')

    for pp in ppb.build_peptides(p_struct):
        start = pp[0].id[1] - 1
        output_str += '-' * (start - prev)
        output_str += pp.get_sequence()
        prev = pp[-1].id[1]
    start = len(seq)
    output_str += '-' * (start - prev)
    print(output_str)
    print(f'-----------')
    for model in p_struct:
        for chain in model:
            residues = Selection.unfold_entities(chain, "R")
            print(residues)
            # for residue in chain:
                # print(residue.id)
    # print(f'MMCIF PARSER ------------------------')
    #
    # print(f'**********')





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






def get_structs(names, display_first=True):
    """
    Gets the structures from the names list.
    Should be stored in the mmCIF_DIR

    = Possible extensions =
    - allow for input of different annotations. "artifact", mutation, etc.
    - multi-sequence allignment? Perform an allignment to see if these structure motifs already exist in wild
    - ADD OUTPUT OF SEQUENCE ANNOTATIONS?? Prediction on where DNA would bind, stuff like that?
            - Info is found on sequence annotations
            - *** Predict under what conditions protein will crystallize best?? (_exptl_crystal_grow.pdbx_details, _exptl_crystal_grow.method, _exptl_crystal_grow.temp)
                - From the key dict: _exptl_crystal_grow.pdbx_details
            - *** Predict what type of method will be best? (_exptl.method) '_exptl.method'
    - Maybe have our output also include molecule details like heteroatom, binding, etc.

    :param names:
    :return:
    """

    # gathers the PDB file for each pdb name in names
    i = 0
    # print(f'name: {names}')
    names = [mmCIF_DIR + "/" + name + ".cif" for name in names]
    pdb_info = MMCIF2Dict(names[1])
    print(f"_entity.id : {pdb_info['_entity.id']}")
    print(f"_entity.type: {pdb_info['_entity.type']}")
    print(f"_entity.src_method: {pdb_info['_entity.src_method']}")
    print(f"_entity.pdbx_description: {pdb_info['_entity.pdbx_description']}")
    print(f"_entity.pdbx_number_of_molecules: {pdb_info['_entity.pdbx_number_of_molecules']}")
    print(f"_entity.details: {pdb_info['_entity.details']}")
    print(f"_entity_poly.entity_id: {pdb_info['_entity_poly.entity_id']}")
    print(f"_entity_poly.nstd_monomer: {pdb_info['_entity_poly.nstd_monomer']}")
    print(f"_entity_poly.type: {pdb_info['_entity_poly.type']}")
    print(f"_entity_poly.pdbx_seq_one_letter_code: {pdb_info['_entity_poly.pdbx_seq_one_letter_code']}")
    print(f"_entity_poly.pdbx_seq_one_letter_code_can: {pdb_info['_entity_poly.pdbx_seq_one_letter_code_can']}")
    print(f"_entity_poly.pdbx_strand_id: {pdb_info['_entity_poly.pdbx_strand_id']}")
    print(f"_entity_poly.pdbx_target_identifier: {pdb_info['_entity_poly.pdbx_target_identifier']}")
    print(f"_entity_poly_seq.entity_id: {pdb_info['_entity_poly_seq.entity_id']}")
    print(f"_entity_poly_seq.num: {pdb_info['_entity_poly_seq.num']}")
    print(f"_entity_poly_seq.mon_id: {pdb_info['_entity_poly_seq.mon_id']}")
    print(f"_entity_poly_seq.hetero: {pdb_info['_entity_poly_seq.hetero']}")
    print(f"_entity.details: {pdb_info['_entity.details']}")
    print(f"_entity_poly.pdbx_target_identifier : {pdb_info['_entity_poly.pdbx_target_identifier']}")
    print(f"_struct_ref.pdbx_seq_one_letter_code {pdb_info['_struct_ref.pdbx_seq_one_letter_code']}")
    print(f"_struct_ref_seq.pdbx_strand_id: {pdb_info['_struct_ref_seq.pdbx_strand_id']}")
    print(f"_struct_ref_seq.seq_align_beg: {pdb_info['_struct_ref_seq.seq_align_beg']}")


    print()

    print(f"_entity_src_gen.entity_id: {pdb_info['_entity_src_gen.entity_id']}")
    print(f"_entity_src_gen.pdbx_src_id: {pdb_info['_entity_src_gen.pdbx_src_id']}")
    print(f"_entity.id : {pdb_info['_entity.id']}")
    print(f"_entity_poly.type: {pdb_info['_entity_poly.type']}")
    print(f"_entity_poly.pdbx_strand_id: {pdb_info['_entity_poly.pdbx_strand_id']}")
    print(f"_struct_ref.pdbx_seq_one_letter_code {pdb_info['_struct_ref.pdbx_seq_one_letter_code']}")
    print(f"_struct_ref_seq.pdbx_strand_id: {pdb_info['_struct_ref_seq.pdbx_strand_id']}")
    print()
    print(f"_pdbx_poly_seq_scheme.pdb_seq_num: {pdb_info['_pdbx_poly_seq_scheme.pdb_seq_num']}")
    print(f"_pdbx_poly_seq_scheme.asym_id: {pdb_info['_pdbx_poly_seq_scheme.asym_id']}")

    print()
    print(f'sequence:')
    for i_str in pdb_info['_entity_src_gen.entity_id']:
        print(f'i id: {str(i_str)}')

    for pep_id in pdb_info['_entity_poly.entity_id']:
        id = int(pep_id) - 1
        type = pdb_info['_entity_poly.type'][id]
        print(f"type: {type}")
        chain_letter = pdb_info['_struct_ref_seq.pdbx_strand_id'][id]
        print(f"chain: {chain_letter}")
        if type == 'polypeptide(L)':
            print(f'**************************************************')
            print(f'SLAYYYYY WE HAVE A POLYPEPTIDE!!!')

    # Goal: get the type of the polomer (protein, not DNA), then cross-reference with
    # cif parser to get the final sequence.

    # to investigate:
    # _struct_ref.pdbx_seq_one_letter_code
    # _struct_ref_seq.pdbx_strand_id
    # _struct_ref_seq.seq_align_beg
    # _struct_ref_seq.db_align_beg
    # _struct_ref_seq.pdbx_auth_seq_align_beg
    # _struct_ref_seq.ref_id

    print(f"_struct_ref.pdbx_seq_one_letter_code: {pdb_info['_struct_ref.pdbx_seq_one_letter_code']}")
    print(f"_struct_ref_seq.pdbx_strand_id: {pdb_info['_struct_ref_seq.pdbx_strand_id']}")
    print(f"_struct_ref_seq.seq_align_beg: {pdb_info['_struct_ref_seq.seq_align_beg']}")
    print(f"_struct_ref_seq.db_align_beg: {pdb_info['_struct_ref_seq.db_align_beg']}")
    print(f"_struct_ref_seq.pdbx_auth_seq_align_beg: {pdb_info['_struct_ref_seq.pdbx_auth_seq_align_beg']}")
    print(f"_struct_ref_seq.ref_id: {pdb_info['_struct_ref_seq.ref_id']}")
    print(f"_struct_ref.pdbx_align_begin: {pdb_info['_struct_ref.pdbx_align_begin']}")
    print(f"")
    print(f"")
    print(f"")


    # -------------------------- SOLUTION!!!!!! ------------------------------
    # _entity_poly.type:
    # _entity_poly.pdbx_seq_one_letter_code: Maybe don't need to use this one?
    # _entity_poly.pdbx_strand_id: ['A,C', 'B,E', 'D,F'] (D,F)
    # _pdbx_poly_seq_scheme.ndb_seq_num
    # _pdbx_poly_seq_scheme.seq_id
    #

    # for key in pdb_info:
    #     print(f'key: {key}')
    # print(f'pdb info: {pdb_info}')
    # print(f'---')
    files = [open(name) for name in names]

    dist_mats = []
    for pid_file, name in zip(files, names):
        p_struct = parser.get_structure(pid_file, pid_file)
        s = ''

        pdb_info = MMCIF2Dict(name)
        peptides = []
        for pep_id in pdb_info['_entity_poly.entity_id']:
            id = int(pep_id) - 1
            type = pdb_info['_entity_poly.type'][id]
            # print(f"type: {type}")
            chain_letter = pdb_info['_struct_ref_seq.pdbx_strand_id'][id]
            # print(f"chain: {chain_letter}")
            if type == 'polypeptide(L)':
                # print(f'**************************************************')
                # print(f'SLAYYYYY WE HAVE A POLYPEPTIDE!!!')
                # print(f"start seq: {pdb_info['_struct_ref.pdbx_align_begin'][id]}")
                s += '-' * (int(pdb_info['_struct_ref.pdbx_align_begin'][id]) - 1)
                peptides.append(chain_letter)
                s += pdb_info['_entity_poly.pdbx_seq_one_letter_code'][id].replace('\n', '')



        # for record in SeqIO.parse(name, "cif-seqres"):
        print(f'*** Records:')
        # record_dict = SeqIO.to_dict(SeqIO.parse(name, "cif-seqres"))
        # print(f'record dict: {record_dict}')
        for index, record in enumerate(SeqIO.parse(name, "cif-seqres")):
            print(record)

        for record in SeqIO.parse(name, "cif-seqres"):



            print("Record id %s, chain %s" % (record.id, record.annotations["chain"]))
            print(f'record annotations: {record.annotations}')
            # print(record.seq[0])

            # print(record.dbxrefs)
            print(record.seq)
            # Just for some data exploration!
            # print(dir(record))
            # print(f'dbxrefs: {record.dbxrefs}')
            # print(f'sequence: {record.seq}')
            # print(f'count: {record.count}')
            # print(f'description: {record.description}')
            # print(f'features: {record.features}')
            # print(f'format: {record.format}')
            # print(f'id: {record.id}')
            # print(f'letter annotations: {record.letter_annotations}')
            # print(f'name: {record.name}')

            print(f'sequence 2: {record.format("embl")}')
            # embl, stockholm, fasta, fasta-2line
            #
            # working: clustal, embl, fasta, fasta-2line, gb, imgt, nexus, phylip, pir, tab
            # helpful: clustal, ***embl***, fasta, imgt



            # if record.seq == 'XXXXXXXXXXXXXXXX':
            #     print(f'NOT adding record: {record}')
            # else:
            #     print(f'adding record!! {record}')
            #     s += record.seq


            # if record.annotations["chain"] in peptides:
            #     print(f'adding to chain...')
            #     print(record)
            #     s += record.seq

            print()


        print(f'-----------')
        polypeptides = sequence_construction(p_struct, s)
        dist_mat = calc_dist_mat(p_struct)
        dist_mats.append(dist_mat)

    plt.figure(figsize=(10, 8))
    sns.heatmap(dist_mats[0], cmap="viridis", square=True, cbar_kws={'label': 'Distance (A)'})
    plt.title(f'Contact Map / Distance matrix for {files[0].name}')


def obtain_training():
    """
    Obtains the training data from our PDB list!
    :return:
    """
    names = download_list()
    dist_mat = get_structs(names[0:6])
    plt.show()




if __name__ == '__main__':
    obtain_training()
    # names = download_list()
    # dist_mat = get_structs(names[0:1])
    # plt.show()


    # Notable proteins:
    # 6L6Y
