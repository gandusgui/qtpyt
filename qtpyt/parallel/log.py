import logging


def get_logger(name, filename="file.log", level=logging.DEBUG, mode="w", open_log=True):
    log = logging.getLogger(name)
    log.setLevel(level)

    fmt = logging.Formatter("%(asctime)s - %(message)s")

    if open_log:
        fh = logging.FileHandler(filename, mode)
    else:
        fh = logging.NullHandler()

    fh.setLevel(level)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log

