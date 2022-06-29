import logging
import datetime
import tqdm
import sys


class TqdmLoggingHandler(logging.Handler):
    # get from this post:
    # https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name, args):
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    if args.load_pretrained_model:
        split_dest_dir = args.dest_dir.split('/')
        logger_name_w_time = split_dest_dir[-1]
        handler = logging.FileHandler('{}/{}.log'.format(args.dest_dir, logger_name_w_time), mode='w')
    else:
        dest_dir = args.dest_dir
        handler = logging.FileHandler('{}/{}_{}.log'.format(dest_dir, now, name), mode='w')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(TqdmLoggingHandler())
    logger.propagate = False
    return logger