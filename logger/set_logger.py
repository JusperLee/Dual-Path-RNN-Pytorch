import logging
from datetime import datetime
import os
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def setup_logger(logger_name, root, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    os.makedirs(root,exist_ok=True)
    if tofile:
        log_file = os.path.join(root, '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


if __name__ == "__main__":
    setup_logger('base','root',level=logging.INFO,screen=True, tofile=False)
    logger = logging.getLogger('base')
    logger.info('hello')
