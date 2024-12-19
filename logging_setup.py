import logging
import logging.handlers
from multiprocessing_logging import install_mp_handler
import os

# === Helpful tips and tricks when setting up logging! ===
# - Make sure you're logging with logger.info(...), NOT logging.info(...). The latter will use the global logger, not ideal.
# - Be careful with import statements. You set up the logger after you run Transformer.py, so when you first call your imports,
#           your python scripts might not have a fully set up logger yet!

change_code = False
logger = logging.getLogger("protein_logger")
logger.setLevel(logging.DEBUG)


class ProteinLogFilter(logging.Filter):
    """
    Stores the name of the current protein code so it can easily be logged.
    """
    def __init__(self):
        super().__init__()
        self.protein_code = '    '
        self.override = False

    def change_protein_code(self, code):
        self.protein_code = code.upper()

    def remove_code(self):
        """
        Makes protein code a default value
        :return:
        """
        self.protein_code = '    '

    def filter(self, record):
        record.protein_code = self.protein_code
        return True


def get_protein_filter(logger=None):
    logger = logger or get_logger()
    for f in logger.filters:
        if isinstance(f, ProteinLogFilter):
            return f
    return None


# Set up a shared logger
def setup_logger(node_name='DEFAULT'):
    # Setup directory for logs to live
    os.makedirs(f'Logs', exist_ok=True)
    os.makedirs(f'Logs/{node_name}', exist_ok=True)
    logger = logging.getLogger("protein_logger")

    # ---------------------- Logging framework ----------------------
    # 10MB exhaustive handlers
    # file_handler = logging.handlers.RotatingFileHandler(f'Logs/{node_name}/Full_Log.log', maxBytes=10000000,
    #                                                     backupCount=5)
    # file_handler.doRollover()
    if node_name == 'DEFAULT':
        file_handler = logging.FileHandler(f'Logs/Full_Log.log', mode='w')
    else:
        file_handler = logging.FileHandler(f'Logs/{node_name}/Full_Log.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    # Starts each call in a new log!

    if node_name == 'DEFAULT':
        master_handler = logging.FileHandler(f'Logs/ERRORS.log', mode='w')
    else:
        master_handler = logging.FileHandler(f'Logs/{node_name}/ERRORS.log', mode='w')
    master_handler.setLevel(logging.WARNING)

    # logging.basicConfig(level=logging.DEBUG, handlers=tuple([master_handler, file_handler]),
    #                     format='%(levelname)-8s: %(asctime)-22s %(module)-15s %(protein_code)-4s - %(message)s',
    #                     datefmt='%Y-%m-%d %H:%M:%S | ')


    logging.basicConfig(level=logging.DEBUG, handlers=(file_handler,master_handler),
                        format='%(levelname)-8s: %(asctime)-22s %(module)-15s %(protein_code)-4s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S | ')
    # Create and return the logger
    print(f'handlers: {logger.handlers}')
    print(f'filters {logger.filters}')
    print(f'logger: {logger}')

    # Add the ProteinFilter if it's not already present
    if not any(isinstance(f, ProteinLogFilter) for f in logger.filters):
        protein_filter = ProteinLogFilter()
        logger.addFilter(protein_filter)
    install_mp_handler(logger)
    print(f'filters: {logger.filters}')
    print(f'logger: {logger}')
    print(f'handlers: {logger.handlers}')
    print(f'SETTING UP!!!!')
    logger.warning(f'first log')
    return logger


def get_logger():
    """Retrieve the logger for modules."""
    logger = logging.getLogger("protein_logger")
    # if not any(isinstance(f, ProteinLogFilter) for f in logger.filters):
    #     protein_filter = ProteinLogFilter()
    #     logger.addFilter(protein_filter)
    #     print(f'adding new filter???')
    # print(f'filters: {logger.filters}')
    # print(f'getting logger')
    # print(f'logger: {logger}')
    # print(f'handlers: {logger.handlers}')
    # print(f'parent: {logger.parent}')
    # print(f'caller: {logger.findCaller(stack_info=True)}')
    # print()
    return logger


def enable_mp():
    """
    Enables handling for multiprocessing
    :return:
    """
    global change_code
    change_code = False
    print(f'turning off change code...')

def disable_mp():
    """
    Disables handling for multiprocessing
    :return:
    """
    global change_code
    change_code = True


def change_log_code(code='    '):
    """
    Changes the 4-letter protein code for the global logger
    :param code:
    :return:
    """
    # global change_code
    # global started
    # print(f'wants to change log code...')
    # print(f'change code: {change_code}')
    # print(f'started? {started}')
    if change_code:
        get_protein_filter(get_logger()).change_protein_code(code)
    # get_protein_filter(get_logger()).change_protein_code(code)
