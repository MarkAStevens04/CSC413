import get_pdbs
import logging
import logging.handlers
import amino_expert
import numpy as np


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)



class Sequence_Parser():
    """
    Class for parsing protein sequences and cleaning up data
    """
    def __init__(self, max_samples=100):
        """ Initializes our sequence parser.

        This is the main source of modifying constants.
        We take from our protein-ids.txt file and parse the proteins in order, up to a maximum of max_samples.

        :type max_samples: int
        :param max_samples: Maximum number of proteins to parse
        """
        # Directory where we store our processed data binaries
        self.bin_dir = 'PDBs/processed_data/'

        self.aa_codes = get_pdbs.download_list(max_download=max_samples)

        # Initialize our amino acid expert
        self.e = amino_expert.AA_Expert()
        self.e.use_ideal_aminos()


    def parse_names(self):
        """ Turns 4-letter protein codes into input & target data.

        Takes each protein code in self.aa_codes, interprets the data, and saves binary to directory in self.bin_dir

        For info on the format of these files, see process_aminos

        :rtype: None
        :return: None. Saves file to processed data directory

        """
        # Sequence info is [protein 1, protein 2, ...] where protein 1 = (4-letter protein code, reference protein sequence, defined_position protein sequence, list of atoms)
        # NOTE: Length of self.aa_codes may NOT equal length of sequence info. Some proteins are culled.
        sequence_info = get_pdbs.get_structs(self.aa_codes)
        logger.info(f'------------------------------------------------------------------------')
        logger.info(f'beginning protein processing!')

        for n in sequence_info:
            # n is (protein code, ref_sequence, position_defined sequence, [list of aminos])

            # n[0] is actually directory to mmCif file. We extract the last 4 characters before the extension.
            self.code = n[0][-8:-4]
            # Store the reference and positional sequences
            self.ref_seq = n[1]
            self.pos_seq = n[2]

            # Process our list of aminos
            # amino has format [amino position, amino 1-letter abbrev, (atom1), (atom2), (atom3), ...]
            # where (atom) = (atom type, [atom x, atom y, atom z])
            given, target = self.process_aminos(n[3])

            np.save(f'{self.bin_dir}{self.code}-in', given)
            np.save(f'{self.bin_dir}{self.code}-target', target)
            logging.info(f'Successfully parsed {self.code}!')


    def process_aminos(self, amino_list) -> (np.array, np.array):
        """ Takes a list of atoms in amino acid and creates abbreviated numpy arrays.

        Processes a list of amino acids!
        Iterates through each amino acid, creates two lists: given and target.

        = DANGER! =
         - Assumes amino_list is SORTED! Should be sorted in standard format.
         - Assumes provided aminos never have more atoms than the standard aminos do.


        = For future =
         - Possibly remove hydrogen positions, as it is very infrequent for these to be actually included in an mmCif file.
         - Clean up initial sequence explicitly by removing '-' from reference sequence

        :param amino_list: List of aminos, each containing list of atoms with some meta-data. Has format: [[amino position, 1-letter abbrev, (atom 1), (atom 2), ...], ...] where (atom1) = (atom code, [x, y, z])
        :return: given and target list.
        Given has format [[atom name index, u, v, w, amino name index], ...].
        u, v, and w are "expert" positions. That is to say, they are the typical positions of each atom in the amino acid.
        Target has format [atom name index, x, y, z, known_position flag].
        x, y, and z are the known to be true positions of each atom. Our known_position flag is a 0 if the position is unknown, and x = y = z = -1, and x, y, and z take on the correct values if the flag is 1 (meaning the position is known).

        """

        # aa_idx will track the current index of our amino acid
        aa_idx = 0
        # processed_given and processed_target will store a list of our atoms
        processed_given = []
        processed_target = []
        # for every letter in our sequence
        for idx, a, s in zip(range(len(self.ref_seq)), self.pos_seq, self.ref_seq):

            # Check whether this amino acid has a known position or not.
            if a != '-':
                # We know that this amino has a known position!

                # Set our current amino to the current one from our amino list
                amino = amino_list[aa_idx]
                aa_idx += 1
                # will slide through every atom. i will track the provided amino, j will track the standard amino.
                i = 0
                j = 0
                # Want to search through every possible atom in the amino acid.
                # Only possible for current amino to be shorter. In this case, we
                # substitute unknown atoms with that flag.

                # while we have not searched through every atom in the standard amino acid...
                while j < len(self.e.aminos[a]):

                    aa_atom = self.e.aminos[a][j]
                    # Second line converts our letter for amino acid into an index
                    # atom_given = [*aa_atom, a]
                    atom_given = [*aa_atom, ord(a.lower()) - 97]
                    if i + 2 >= len(amino):
                        # We know we have searched through every atom in the given amino!
                        # Still append the atom from standard amino to the target, but with undefined positions.
                        atom_target = [aa_atom[0], -1, -1, -1, 0]
                        processed_given.append(atom_given)
                        processed_target.append(atom_target)
                        j += 1
                    else:
                        # Check the current atom in our given amino. (i + 2 because first 2 indices store amino position and name, respectively)
                        # Curr atom has format (atom name string, [x, y, z])
                        curr_atom = amino[i + 2]
                        curr_atom_label = self.e.encode[curr_atom[0]]

                        # Check if the given atom is the same as the standard amino atom
                        if int(aa_atom[0]) == curr_atom_label:
                            # Our current atom is the same as the standard amino atom!
                            # We slide our frame for both our given amino and standard amino.

                            # Our target has a known position
                            atom_target = [aa_atom[0], *curr_atom[1], 1]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)

                            i += 1
                            j += 1

                        elif curr_atom[0] not in self.e.encode:
                            # If the name of the current atom has never been seen in our standard aminos...
                            # We know it is likely a nonstandard atom, and we will NOT add it to our list!

                            logger.warning(f'nonstandard atom! {curr_atom[0]}, amino: {amino}')
                            print(f'nonstandard atom! {curr_atom[0]}, amino: {amino}')
                            # Do not add to target!
                            i += 1
                        else:
                            # We know that the current atom in given and current atom in the standard are not a match.
                            # We assume that given amino's atoms are a subset of the standard amino's atoms.
                            # So we know that this standard atom is not in the given amino.
                            # We add the atom to our target list with an undefined position.
                            # If there's no match, we will continue scanning in the amino acid.

                            atom_target = [aa_atom[0], -1, -1, -1, 0]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)
                            j += 1

                # Error checking. Make sure every atom from our amino has been inserted.
                # This error will occur if there is an atom out-of-order, or if for some reason
                # the given amino's atoms are not a subset of the standard amino's atoms.
                if i != len(amino) - 2:
                    print(f'BOUNDS ERROR!!!! {len(amino)}, {i}')
                    logger.warning(f'BOUNDS ERROR!!! Possibly invalid protein! Length of given amino: {len(amino) - 2}, got to {i}, amino: {amino}')

            elif s != '-':
                # We know the amino does NOT have any defined position!
                # But it is a valid amino in our sequence
                j = 0
                while j < len(self.e.aminos[s]):
                    # We will iterate through each atom in our standard amino.
                    # Add each atom to both given and target lists, but with
                    # no defined position in our target.

                    aa_atom = self.e.aminos[s][j]

                    atom_given = [*aa_atom, ord(s.lower()) - 97]
                    atom_target = [aa_atom[0], -1, -1, -1, 0]

                    processed_given.append(atom_given)
                    processed_target.append(atom_target)
                    j += 1

        # Turn our atom lists into numpy arrays.
        # This will allow us to store the binaries of the numpy arrays and access them via
        # memory mapping.
        input = np.array(processed_given, dtype='f')
        output = np.array(processed_target, dtype='f')

        return input, output

    def open_struct(self, name):
        """ Opens structure with 4-letter protein code.

        Uses the directory at self.bin_dir to find the binary files of our stored arrays.

        Opens the file into a given and target file.

        :param name: 4 letter code where we store our binary
        :return:
        """

        given = np.load(f'{self.bin_dir}{name}-in.npy', mmap_mode='r', allow_pickle=True)
        target = np.load(f'{self.bin_dir}{name}-target.npy', mmap_mode='r', allow_pickle=True)

        print(f'train: {given}')
        print()
        print(f'target: {target}')
        print()










if __name__ == '__main__':

    # ---------------------- Logging framework ----------------------
    # 10MB handlers
    file_handler = logging.handlers.RotatingFileHandler('Logs/Full_Log.log', maxBytes=10000000,
                                                        backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    # Starts each call as a new log!
    file_handler.doRollover()

    master_handler = logging.FileHandler('Logs/WARNINGS.log', mode='w')
    master_handler.setLevel(logging.WARNING)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, master_handler],
                        format='%(levelname)-8s: %(asctime)-22s %(module)-20s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S | ')

    logger.info('Started!')
    # ---------------------- End Logging Framework ----------------------

    print(f'parsing')
    a = Sequence_Parser()
    a.parse_names()
    # a.open_struct('6L6Y')
