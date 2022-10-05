import inspect
import logging
import functools
import threading
from os import environ
from config.config import ConfigFactory

# Get configs
configs = ConfigFactory.factory(environ).configs


def create_logger():
    """
    Creates a logging object and returns it
    """
    _logger = logging.getLogger("example_logger")
    _logger.setLevel(logging.INFO)
    # create the logging file handler
    log_path = configs['Log']['params']['log_path']
    fh = logging.FileHandler(log_path)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    _logger.addHandler(fh)
    return _logger


# create logger
logger = create_logger()


def exception(function):
    """
    A decorator that wraps the passed in function and logs
    exceptions should one occur
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except KeyboardInterrupt:
            err = "KeyboardInterrupt"
            logger.info(err)
            raise
        except:
            # get module name
            frm = inspect.trace()[-1]
            mod = inspect.getmodule(frm[0])
            modname = mod.__name__ if mod else frm[1]
            # log the exception
            err = f"{threading.current_thread().name} : There was an exception in "
            err += function.__name__
            err += f', module {modname}.'
            logger.exception(err)
            # re-raise the exception
            # raise
    return wrapper
