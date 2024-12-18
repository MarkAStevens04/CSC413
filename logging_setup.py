import logging
import logging.handlers
import os


class ProteinLogFilter(logging.Filter):
    """
    Stores the name of the current protein code so it can easily be logged.
    """
    def __init__(self):
        super().__init__()
        self.protein_code = '----'

    def change_protein_code(self, code):
        self.protein_code = code.upper()

    def remove_code(self):
        """
        Makes protein code a default value
        :return:
        """
        self.protein_code = '----'

    def filter(self, record):
        record.protein_code = self.protein_code
        return True


def get_protein_filter(logger=None):
    logger = logger or logging.getLogger("protein_logger")
    for f in logger.filters:
        if isinstance(f, ProteinLogFilter):
            return f
    return None

# Set up a shared logger
def setup_logger(node_name='DEFAULT'):
    # Setup directory for logs to live
    os.makedirs(f'Logs', exist_ok=True)
    os.makedirs(f'Logs/{node_name}', exist_ok=True)

    # ---------------------- Logging framework ----------------------
    # 10MB exhaustive handlers
    file_handler = logging.handlers.RotatingFileHandler(f'Logs/{node_name}/Full_Log.log', maxBytes=10000000,
                                                        backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    # Starts each call in a new log!
    file_handler.doRollover()

    master_handler = logging.FileHandler(f'Logs/{node_name}/ERRORS.log', mode='w')
    master_handler.setLevel(logging.WARNING)

    logging.basicConfig(level=logging.DEBUG, handlers=[master_handler, file_handler],
                        format='%(levelname)-8s: %(asctime)-22s %(module)-20s %(protein_code)-4s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S | ')
    # Create and return the logger
    logger = logging.getLogger("protein_logger")
    # Add the ProteinFilter if it's not already present
    if not any(isinstance(f, ProteinLogFilter) for f in logger.filters):
        protein_filter = ProteinLogFilter()
        logger.addFilter(protein_filter)
    return logger


def get_logger():
    """Retrieve the logger for modules."""
    return logging.getLogger("protein_logger")


def change_code(code='----'):
    get_protein_filter(get_logger()).change_protein_code(code)