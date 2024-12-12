import get_pdbs
import logging
import logging.handlers
import amino_expert
import numpy as np
import time
import os

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


    def parse_names(self, aa_codes=None):
        """ Turns 4-letter protein codes into input & target data.

        Takes each protein code in self.aa_codes, interprets the data, and saves binary to directory in self.bin_dir

        For info on the format of these files, see process_aminos

        :param aa_codes: Defaults to using self.aa_codes
        :rtype: None
        :return: None. Saves file to processed data directory

        """
        # Sequence info is [protein 1, protein 2, ...] where protein 1 = (4-letter protein code, reference protein sequence, defined_position protein sequence, list of atoms)
        # NOTE: Length of self.aa_codes may NOT equal length of sequence info. Some proteins are culled.
        # Default to using our own codes, otherwise use codes provided.
        if aa_codes is None:
            sequence_info = get_pdbs.get_structs(self.aa_codes)
        else:
            sequence_info = get_pdbs.get_structs(aa_codes)
        logger.info(f'------------------------------------------------------------------------')
        logger.info(f'beginning protein processing!')

        for n in sequence_info:
            # n is (protein code, ref_sequence, position_defined sequence, [list of aminos])

            # n[0] is actually directory to mmCif file. We extract the last 4 characters before the extension.
            code = n[0][-8:-4]
            # Store the reference and positional sequences
            ref_seq = n[1]
            pos_seq = n[2]

            # Process our list of aminos
            # amino has format [amino position, amino 1-letter abbrev, (atom1), (atom2), (atom3), ...]
            # where (atom) = (atom type, [atom x, atom y, atom z])
            logging.debug(f'Parsing {code}...')
            given, target = self.process_aminos(ref_seq, pos_seq, n[3])

            # Check for our error signal
            if given is None:
                logging.error(f'Throwing away protein {code}')
            else:
                new_ref = ref_seq.replace('-', '')
                # Verify that our sequence length is what we expect
                if (len(new_ref) * 27) == target.shape[0]:
                    np.save(f'{self.bin_dir}{code}-in', new_ref)
                    np.save(f'{self.bin_dir}{code}-target', target)
                    # print(f'target: {target[54:150, :]}')
                    logging.info(f'Successfully parsed {code}!')
                else:
                    logging.warning(f'len ref_seq * 27 != target.shape[0]! ')
                    logging.warning(f'mismatch between actual & expected length')
                    logging.error(f'Throwing away protein {code}')


                # print(f"modi ref seq: {new_ref}")
                # print(f"verify seque: {len(new_ref) * 27}, {target.shape[0]}")
                # np.save(f'')



    def process_aminos(self, ref_seq, pos_seq, amino_list) -> (np.array, np.array):
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
        # Begin by extracting the largest number of atoms in an amino acid (we find its 27)
        lengths = [len(self.e.aminos[a]) for a in self.e.aminos]
        max_size = max(lengths)


        # aa_idx will track the current index of our amino acid
        aa_idx = 0
        # processed_given and processed_target will store a list of our atoms
        processed_given = []
        processed_target = []
        # for every letter in our sequence
        for idx, a, s in zip(range(len(ref_seq)), pos_seq, ref_seq):
            num_added = 0
            # Check whether this amino acid has a known position or not.
            if a != '-':
                # We know that this amino has a known position!

                # Set our current amino to the current one from our amino list
                amino = amino_list[aa_idx]
                aa_idx += 1
                # will slide through every atom. i will track the provided amino, j will track the standard amino.
                i = 0
                j = 0


                # Check if the current amino acid abbreviation is valid. If not, throw error!
                if a not in self.e.aminos:
                    logger.warning(f"Invalid amino name! '{a}'")
                    return None, None

                # Want to search through every possible atom in the amino acid.
                # Only possible for current amino to be shorter. In this case, we
                # substitute unknown atoms with that flag.

                # while we have not searched through every atom in the standard amino acid...
                while j < len(self.e.aminos[a]):

                    # print(f'amino codes: {[ord(a.lower()) - 97 for a in self.e.aminos.keys()]}')

                    aa_atom = self.e.aminos[a][j]
                    # Second line converts our letter for amino acid into an index
                    # atom_given = [*aa_atom, a]
                    atom_given = [*aa_atom, ord(a.lower()) - 97]
                    if i + 2 >= len(amino):
                        # We know we have searched through every atom in the given amino!
                        # Still append the atom from standard amino to the target, but with undefined positions.
                        atom_target = [aa_atom[0], -1, -1, -1, 0, 1]
                        processed_given.append(atom_given)
                        processed_target.append(atom_target)
                        num_added += 1
                        j += 1
                    else:
                        # Check the current atom in our given amino. (i + 2 because first 2 indices store amino position and name, respectively)
                        # Curr atom has format (atom name string, [x, y, z])
                        curr_atom = amino[i + 2]

                        if curr_atom[0] not in self.e.encode:
                            # If the name of the current atom has never been seen in our standard aminos...
                            # We know it is likely a nonstandard atom, and we will NOT add it to our list!

                            logger.warning(f'nonstandard atom! {curr_atom[0]}, amino: {amino}')
                            print(f'nonstandard atom! {curr_atom[0]}, amino: {amino}')
                            # Do not add to target!
                            i += 1
                        elif int(aa_atom[0]) == self.e.encode[curr_atom[0]]:
                            # Check if the given atom is the same as the standard amino atom
                            # Our current atom is the same as the standard amino atom!
                            # We slide our frame for both our given amino and standard amino.

                            # Our target has a known position
                            atom_target = [aa_atom[0], *curr_atom[1], 1, 1]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)
                            num_added += 1

                            i += 1
                            j += 1

                        else:
                            # We know that the current atom in given and current atom in the standard are not a match.
                            # We assume that given amino's atoms are a subset of the standard amino's atoms.
                            # So we know that this standard atom is not in the given amino.
                            # We add the atom to our target list with an undefined position.
                            # If there's no match, we will continue scanning in the amino acid.

                            atom_target = [aa_atom[0], -1, -1, -1, 0, 1]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)
                            num_added += 1
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

                # Check if the reference sequence has a known position
                if s not in self.e.aminos:
                    logger.warning(f'Invalid amino name! {a}')
                    return None, None


                while j < len(self.e.aminos[s]):
                    # We will iterate through each atom in our standard amino.
                    # Add each atom to both given and target lists, but with
                    # no defined position in our target.

                    aa_atom = self.e.aminos[s][j]

                    atom_given = [*aa_atom, ord(s.lower()) - 97]
                    atom_target = [aa_atom[0], -1, -1, -1, 0, 1]

                    processed_given.append(atom_given)
                    processed_target.append(atom_target)
                    num_added += 1
                    j += 1

            if s != '-':
                # Add blank atoms to make each amino produce the same number of atom output
                # Only do this when the referenced sequence is not blank
                while num_added < max_size:
                    processed_given.append([0, 0, 0, 0, 27])
                    processed_target.append([0, 0, 0, 0, 0, 0])
                    num_added += 1


        # Turn our atom lists into numpy arrays.
        # This will allow us to store the binaries of the numpy arrays and access them via
        # memory mapping.
        input = np.array(processed_given, dtype='f')
        output = np.array(processed_target, dtype='f')


        return input, output



    def RAM_Efficient_parsing(self, batch_size=1000):
        """
        Parses params in batches. Slightly slower but much more RAM efficient.
        Use if running into memory problems.
        :param batch_size:
        :return:
        """
        print(f'remainder: {len(self.aa_codes) % batch_size}')
        print(f'adding: {(batch_size - (len(self.aa_codes) % batch_size)) % batch_size}')
        to_split = self.aa_codes + [''] * ((batch_size - (len(self.aa_codes) % batch_size)) % batch_size)
        to_split = np.array(to_split)
        to_split = np.reshape(to_split, (-1, batch_size))
        print(f'to split: {to_split}')
        for t, names in enumerate(to_split):
            logger.info(f'------------------------ started parsing {names} ------------------------')
            percent = (t / to_split.shape[0]) * 100
            print(f'parsing: {names} {(round(percent, 1))}% there!!')
            self.parse_names(names)




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
        print(f'{target.shape()}')
        print(f'target: {target[50:150, -5:]}')
        print()
        largest = 0
        smallest = 10000000000

        # for n in os.listdir('PDBs/processed_data'):
        #     given = np.load(f'PDBs/processed_data/{n}', mmap_mode='r', allow_pickle=True)
        #     largest = max(largest, given.shape[0])
        #     smallest = min(smallest, given.shape[0])
        #     print(f'{given.shape}, {smallest}, {largest}, {n}')








if __name__ == '__main__':

    # ---------------------- Logging framework ----------------------
    # 10MB handlers
    file_handler = logging.handlers.RotatingFileHandler('Logs/Full_Log.log', maxBytes=10000000,
                                                        backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    # Starts each call as a new log!
    file_handler.doRollover()

    master_handler = logging.FileHandler('Logs/ERRORS.log', mode='w')
    master_handler.setLevel(logging.ERROR)

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, master_handler],
                        format='%(levelname)-8s: %(asctime)-22s %(module)-20s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S | ')

    logger.info('Started!')
    logger.warning(f'seeing if this is working')
    # ---------------------- End Logging Framework ----------------------

    print(f'parsing')
    start = time.time()
    a = Sequence_Parser(max_samples=100)
    # a.parse_names(['6XTB'])
    print(a.e.encode)
    a.RAM_Efficient_parsing(batch_size=10)
    # a.open_struct('6XTB')

    logging.info(f'Took {time.time() - start} seconds!!!')
