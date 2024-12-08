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
            self.process_amino(a)
        print()

    def process_amino(self, amino):
        """
        Processes a single amino acid!
        :param amino:
        :return:
        """
        print(f'processing {amino}')
        # print(f'expert: {self.e}')
        # print(self.e.aminos)
        # print(self.e.vocab)

        # print(f'---')
        # print(self.e.encode(['C']))
        for a in self.pos_seq:
            # just iterate through known amino atom positions...
            if a != '-':
                print(f'should be same: {a, amino[1]}')
                print(f'')
                print(self.e.aminos[a])
                print(self.e.vocab)
                for atom in amino[2:]:
                    print(f'atom name: {atom[0]}')
                    print(f'encoding: {atom[0]}:{self.e.encode(atom[0])}')
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

