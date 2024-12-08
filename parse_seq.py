import get_pdbs
import logging
import logging.handlers
import amino_expert



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




class Sequence_Parser():
    """
    Class for parsing protein sequences and cleaning up data
    """
    def __init__(self, max_samples=2):
        self.aa_codes = get_pdbs.download_list(max_download=max_samples)

        self.e = amino_expert.AA_Expert()
        self.e.use_ideal_aminos()

        self.get_structs()


    def get_structs(self):
        """
        Gets all atoms from every sequence!
        :return:
        """
        r = get_pdbs.get_structs(self.aa_codes)
        n = r[0]
        # Store the reference and positional sequences
        self.ref_seq = n[0]
        self.pos_seq = n[1]

        self.full_seq = []
        print(f'***')
        # for n in r:
        print(n[0])
        print(n[1])
        for a in n[2]:
            # amino has format [amino position, amino 1-letter abbrev, (atom1), (atom2), (atom3), ...]
            # where (atomN) = (atom type, [atom x, atom y, atom z])
            self.process_amino(a)
        print()

    def process_amino(self, amino):
        """
        Processes a single amino acid!
        :param amino:
        :return:
        """
        print(f'processing {amino}')

        processed_given = []
        processed_target = []
        # for every letter in our sequence
        for a in self.pos_seq:
            print()
            print()
            print()
            print(f'new letter!!!')
            atom_given = []
            atom_target = []
            # just iterate through known amino atom positions...
            if a != '-':

                # print(f'should be same: {a, amino[1]}')
                # print(f'')
                # print(self.e.aminos[a])
                # print(self.e.vocab)
                # print(f'encode: {self.e.encode}')
                # will slide through every atom.
                i = 0
                j = 0
                end = False
                # Want to search through every possible atom in the amino acid.
                # Only possible for current amino to be shorter. In this case, we
                # substitute unknown atoms with that flag.
                # while i < len(amino) - 2 and j < len(self.e.aminos[a]):
                while j < len(self.e.aminos[a]):
                    # our given atom is the atom of the expert, alongside the name of the residue.
                    aa_atom = self.e.aminos[a][j]
                    # Second line converts our letter for amino acid into an inde
                    atom_given = [*aa_atom, a]
                    # atom_given = [*aa_atom, ord(a.lower()) - 97]
                    if i + 2 >= len(amino):
                        # We know we have searched through every known atom!
                        # Append the atom to the lists still, but with undefined positions.

                        # target atom has unknown position
                        atom_target = [aa_atom[0], -1, -1, -1, 0]
                        processed_given.append(atom_given)
                        processed_target.append(atom_target)
                        j += 1
                    else:
                        # We will search through the current atom
                        curr_atom = amino[i + 2]
                        print(f'curr atom name: {curr_atom[0]}')
                        print(f'encoding: {curr_atom[0]}:{self.e.encode[curr_atom[0]]}')
                        curr_atom_label = self.e.encode[curr_atom[0]]
                        # match = aa_atom[0] == curr_atom_label
                        # print(f'match? amino atom: {aa_atom[0]}, {curr_atom_label}')

                        # Check if the current atom is the same as the supposed amino atom
                        if int(aa_atom[0]) == curr_atom_label:
                            print(f'slay we have a match')
                            # our target has a known position equal to its given position
                            atom_target = [aa_atom[0], *curr_atom[1], 1]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)


                            i += 1
                            j += 1
                        elif curr_atom[0] not in self.e.encode:
                            # if the current atom does not exist in our encoding...
                            # we know it is a nonstandard atom, and we do NOT add it to our list!
                            print(f'nonstandard atom...')
                            # Do not add to target amino!
                            print(f'atom {curr_atom[0]}')
                            i += 1
                        else:
                            # if there's no match, we will continue scanning in the amino acid.
                            # Mark that the position is undefined.
                            print(f'no match :( {aa_atom[0]}, {curr_atom_label}')
                            # target atom has unknown position
                            atom_target = [aa_atom[0], -1, -1, -1, 0]
                            processed_given.append(atom_given)
                            processed_target.append(atom_target)
                            j += 1

                # print()
                # print(f'target:')
                # for l in processed_target:
                #     print(l)

                    # i += 1
                    # j += 1
                # for atom in amino[2:]:
            else:
                # Still want to produce everything we have!
                print(f'not processing this yet...')

            # processed_given.append(atom_given)
            # processed_target.append(atom_target)

        print(f'finished!')
        pause














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
    a.get_structs()

