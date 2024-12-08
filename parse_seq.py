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
    def __init__(self, max_samples=2):
        # Directory where we store our processed data binaries
        self.bin_dir = 'PDBs/processed_data/'

        self.aa_codes = get_pdbs.download_list(max_download=max_samples)

        self.e = amino_expert.AA_Expert()
        self.e.use_ideal_aminos()


    def get_structs(self):
        """
        Gets all atoms from every sequence!
        :return:
        """
        print(f'codes: {self.aa_codes}')
        r = get_pdbs.get_structs(self.aa_codes)
        print(f'len r: {len(r)}')
        for n in r:
            # n is (protein code, ref_sequence, position_defined sequence, [list of atoms])
            print(f'len n: {len(n)}')

            # code = self.aa_codes[]
            # code is whatever our file name is n[0], at the last 4 letters before the extension!
            self.code = n[0][-8:-4]
            # Store the reference and positional sequences
            self.ref_seq = n[1]
            self.pos_seq = n[2]

            self.full_seq = []
            print(f'***')
            # for n in r:
            print(n[0])
            print(n[1])
            print(n[2])
            # print(f'n3: {n[3]}')
            print(f'len n3: {len(n[3])}')
            print(f'len n1: {len(n[1])}')
            for a in n[3]:
                print(a)
            # for a in n[3]:
            #     # amino has format [amino position, amino 1-letter abbrev, (atom1), (atom2), (atom3), ...]
            #     # where (atomN) = (atom type, [atom x, atom y, atom z])
            #     given, target = self.process_amino(a)
            #     print(f'code: {self.code}')
            #     # given.tofile(f'{self.bin_dir}{self.code}.bin')
            #     np.save(f'{self.bin_dir}{self.code}', given)

            # amino has format [amino position, amino 1-letter abbrev, (atom1), (atom2), (atom3), ...]
            # where (atomN) = (atom type, [atom x, atom y, atom z])
            given, target = self.process_aminos(n[3])
            print(f'code: {self.code}')
            # given.tofile(f'{self.bin_dir}{self.code}.bin')
            np.save(f'{self.bin_dir}{self.code}-in', given)
            np.save(f'{self.bin_dir}{self.code}-target', target)
            print()

    def process_aminos(self, amino_list) -> (np.array, np.array):
        """
        Processes a list of amino acids!
        Iterates through each amino acid, creates two lists: given and target.

        :param amino_list: [amino index, 1-letter abbrev, (atom 1), (atom 2), ...] where (atom1) = (atom code, [x, y, z])
        :return:
        """
        # print(f'processing {amino}')

        # aa_idx will track the current index of our amino acid
        aa_idx = 0
        processed_given = []
        processed_target = []
        # for every letter in our sequence
        for idx, a, s in zip(range(len(self.ref_seq)), self.pos_seq, self.ref_seq):
            # print()
            # print()
            # print()
            # print(f'new letter!!!')
            # just iterate through known amino atom positions...
            if a != '-':
                # set our current amino to the current one from our amino list
                amino = amino_list[aa_idx]
                aa_idx += 1
                # will slide through every atom.
                i = 0
                j = 0
                end = False
                # Want to search through every possible atom in the amino acid.
                # Only possible for current amino to be shorter. In this case, we
                # substitute unknown atoms with that flag.
                while j < len(self.e.aminos[a]):
                    # our given atom is the atom of the expert, alongside the name of the residue.
                    aa_atom = self.e.aminos[a][j]
                    # Second line converts our letter for amino acid into an index
                    # atom_given = [*aa_atom, a]
                    atom_given = [*aa_atom, ord(a.lower()) - 97]
                    if i + 2 >= len(amino):
                        # We know we have searched through every known atom!
                        # Append the atom to the lists still, but with undefined positions.
                        # print(f'not included :(')
                        # target atom has unknown position
                        atom_target = [aa_atom[0], -1, -1, -1, 0]
                        processed_given.append(atom_given)
                        processed_target.append(atom_target)
                        j += 1
                    else:
                        # We will search through the current atom
                        curr_atom = amino[i + 2]
                        curr_atom_label = self.e.encode[curr_atom[0]]

                        # Check if the current atom is the same as the supposed amino atom
                        if int(aa_atom[0]) == curr_atom_label:
                            # print(f'slay we have a match')
                            # our target has a known position equal to its given position
                            atom_target = [aa_atom[0], *curr_atom[1], 1]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)

                            i += 1
                            j += 1

                        elif curr_atom[0] not in self.e.encode:
                            # if the current atom does not exist in our encoding...
                            # we know it is a nonstandard atom, and we do NOT add it to our list!
                            logger.warning(f'nonstandard atom! {curr_atom[0]}')
                            print(f'nonstandard atom...')
                            # Do not add to target amino!
                            print(f'atom {curr_atom[0]}')
                            i += 1
                        else:
                            # if there's no match, we will continue scanning in the amino acid.
                            # Mark that the position is undefined.
                            # print(f'no match :( {aa_atom[0]}, {curr_atom_label}')
                            # target atom has unknown position
                            atom_target = [aa_atom[0], -1, -1, -1, 0]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)
                            j += 1

                # Check that we actually inserted all atoms from our amino acid
                if i != len(amino) - 2:
                    print(f'BOUNDS ERROR!!!! {len(amino)}, {i}')
                    logger.warning(f'BOUNDS ERROR!!! Possibly invalid protein! Length of given amino: {len(amino) - 2}, got to {i}, amino: {amino}')

            elif s != '-':
                # We know the atom does NOT have any defined position!
                # But it is a valid atom in our sequence
                j = 0
                while j < len(self.e.aminos[s]):
                    # our given atom is the atom of the expert, alongside the name of the residue.

                    aa_atom = self.e.aminos[s][j]

                    # We know we have searched through every known atom!
                    # Append the atom to the lists still, but with undefined positions.

                    # target atom has unknown position
                    # still regular given atom
                    atom_target = [aa_atom[0], -1, -1, -1, 0]
                    # atom_given = [*aa_atom, s]
                    atom_given = [*aa_atom, ord(s.lower()) - 97]

                    processed_given.append(atom_given)
                    processed_target.append(atom_target)
                    j += 1
                # Still want to produce everything we have!
                # print(f'not processing this yet...')

            # processed_given.append(atom_given)
            # processed_target.append(atom_target)

        # print(f'finished!')
        # print()
        # print()
        # print()
        # print(f'input: ')
        input = np.array(processed_given, dtype='f')
        output = np.array(processed_target, dtype='f')
        # print(input)
        # print(input.shape)

        return input, output

    def open_struct(self, name):
        """
        Opens the structure with the 4 letter code Name
        :param name: 4 letter code where we store our binary
        :return:
        """
        # train = np.memmap(f'{self.bin_dir}{name}.bin', dtype='f', mode='r')
        # print(f'train: {train}')
        given = np.load(f'{self.bin_dir}{name}-in.npy', mmap_mode='r', allow_pickle=True)
        print(f'train: ')
        print(given)

        target = np.load(f'{self.bin_dir}{name}-target.npy', mmap_mode='r', allow_pickle=True)
        print(f'target: ')
        print(target[:100])
        # Sample input
        # [[ 6.200e+01 -1.816e+00  1.420e-01 -1.166e+00  1.200e+01]
        #  [ 1.000e+00 -3.920e-01  4.990e-01 -1.214e+00  1.200e+01]
        #  [ 0.000e+00  2.060e-01  2.000e-03 -2.504e+00  1.200e+01]
        #  ...
        #  [ 2.500e+01 -1.699e+00 -5.070e-01 -9.200e-01  1.800e+01]
        #  [ 4.100e+01 -9.780e-01 -6.790e-01 -3.088e+00  1.800e+01]
        #  [ 5.700e+01 -1.183e+00  1.420e-01  2.828e+00  1.800e+01]]

        # seq: -------------------------------------------------------------GSFTSRIRRPMNAFMVWAKDERKRLAQQNPDLHNAELSKMLGKSWKALTLAEKRPFVEEAKRLRVQHMQDHPNYKYRPRRRKQ
        # out: -------------------------------------------------------------GSFTSRIRRPMNAFMVWAKDERKRLAQQNPDLHNAELSKMLGKSWKALTLAEKRPFVEEAKRLRVQHMQDHPNYKYRPRR---











if __name__ == '__main__':

    # ---------------------- Logging framework ----------------------
    # 10MB handlers
    file_handler = logging.handlers.RotatingFileHandler('Logs/Full_Log.log', maxBytes=10000000,
                                                        backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    # Starts each call as a new log!
    file_handler.doRollover()

    logging.basicConfig(level=logging.DEBUG, handlers=[file_handler],
                        format='%(levelname)-8s: %(asctime)-22s %(module)-20s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S | ')

    logger.info('Started!')
    # ---------------------- End Logging Framework ----------------------

    print(f'parsing')
    a = Sequence_Parser()
    # a.get_structs()
    a.open_struct('6L6Y')
