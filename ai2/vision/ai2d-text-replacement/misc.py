import logging

logger = logging.getLogger(__name__)

def init_logging(log_format='default', log_level='debug'):
    if log_level == 'debug':
        base_logging_level = logging.DEBUG
    elif log_level == 'info':
        base_logging_level = logging.INFO
    elif log_level == 'warning':
        base_logging_level = logging.WARNING
    else:
        raise TypeError('%s is an incorrect logging type!', log_level)
    if len(logger.handlers) == 0:
        ch = logging.StreamHandler()
        logger.setLevel(base_logging_level)
        ch.setLevel(base_logging_level)
        if log_format == 'default':
            formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s: %(message)s \t[%(filename)s: %(lineno)d]', datefmt='%m/%d %I:%M:%S')
        elif log_format == 'defaultMilliseconds':
            formatter = logging.Formatter(fmt='%(asctime)s: %(levelname)s: %(message)s \t[%(filename)s: %(lineno)d]')
        else:
            formatter = logging.Formatter(fmt=log_format, datefmt='%m/%d %I:%M:%S')

        ch.setFormatter(formatter)
        logger.addHandler(ch)
