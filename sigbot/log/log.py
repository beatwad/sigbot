"""
This module provides functions for logging of program activities and
its exceptions.
"""

import functools
import inspect
import logging
import threading
from logging.handlers import RotatingFileHandler
from os import environ, path
from typing import Callable, Union

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
    basedir = path.abspath(path.dirname(__file__))
    log_path = configs["Log"]["params"]["log_path"]
    log_file = f"{basedir}/{log_path}"
    fh = RotatingFileHandler(
        log_file,
        mode="a",
        maxBytes=5 * 1024 * 1024,
        backupCount=2,
        encoding=None,
        delay=False,
    )
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(fmt)
    fh.setFormatter(formatter)
    # add handler to logger object
    _logger.addHandler(fh)
    return _logger


# create logger
logger = create_logger()


def format_exception(function: Union[Callable, None] = None) -> None:
    """
    Format exception text and log it
    """
    # get module name
    frm = inspect.trace()[-1]
    mod = inspect.getmodule(frm[0])
    modname = mod.__name__ if mod else frm[1]
    # log the exception
    err = f"{threading.current_thread().name} : There was an exception in "
    if function is not None:
        err += function.__name__
    err += f", module {modname}."
    logger.exception(err)


def exception(function: Callable):
    """
    A decorator that wraps passed function and logs
    exceptions should one occur
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            func = function(*args, **kwargs)
        except KeyboardInterrupt:
            func = None
            err = "KeyboardInterrupt"
            logger.info(err)
        except BaseException:  # noqa
            func = None
            # format and log exception
            format_exception(function)
        return func

    return wrapper


if __name__ == "__main__":
    logger.info("test")
