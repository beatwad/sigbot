"""
This module provides a configuration system that dynamically loads environment-specific
settings based on the `ENV` environment variable. It includes a `ConfigFactory` class
that returns different configuration objects for various environments
(e.g., development, testing, production, Docker, optimization modes, etc.), and several
`Config` classes defining specific settings for each environment.
"""

import json
from os import path

from dotenv import find_dotenv, load_dotenv

basedir = path.abspath(path.dirname(__file__))
# here we load environment variables from .env, must be called before init. class
load_dotenv(find_dotenv("../.env"), verbose=True)


class ConfigFactory:
    """
    Factory class to return the appropriate configuration settings
    based on the environment variable `ENV`.

    Methods
    -------
    factory(environ: dict) -> Config:
        Returns the appropriate configuration class based on `ENV` value.
    """

    @staticmethod
    def factory(environ):
        """
        Returns the appropriate configuration object depending on the
        environment setting in `ENV`.

        Parameters
        ----------
        environ : _Environ
            A dictionary representing the environment variables.

        Returns
        -------
        Config
            An instance of a configuration class based on the environment.
        """
        env = environ.get("ENV", "development")
        if env == "test":
            return Testing(environ)
        if env == "5m_1h":
            return Development5M1H(environ)
        if env == "15m_1h":
            return Development15M1H(environ)
        if env == "15m_4h":
            return Development15M4H(environ)
        if env == "1h_4h":
            return Development1H4H(environ)
        if env == "optimize":
            return Optimize(environ)
        if env == "docker":
            return Docker(environ)
        if env == "production":
            return Production(environ)
        return Debug(environ)


class Config:
    """
    Base configuration class.

    Attributes
    ----------
    CONFIG_PATH : str
        Path to the configuration file.

    Methods
    -------
    get_config(conf_path: str) -> dict:
        Loads the configuration from the given JSON file.
    """

    CONFIG_PATH = ""

    @staticmethod
    def get_config(conf_path):
        """Load config from JSON file"""
        with open(conf_path) as file:
            bot_conf = json.load(file)
        return bot_conf


class Development(Config):
    """
    Development configuration with basic settings.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.

    Methods
    -------
    __init__(environ: dict):
        Initializes the configuration using environment variables.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the development configuration using the environment variables.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH"))
        self.configs = Config.get_config(pth)


class Development5M1H(Config):
    """
    Development configuration for 5m_1h environment.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the development configuration for the 5m_1h environment.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_5m_1h"))
        self.configs = Config.get_config(pth)


class Development15M1H(Config):
    """
    Development configuration for 15m_1h environment.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the development configuration for the 15m_1h environment.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_15m_1h"))
        self.configs = Config.get_config(pth)


class Development15M4H(Config):
    """
    Development configuration for 15m_4h environment.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the development configuration for the 15m_4h environment.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_15m_4h"))
        self.configs = Config.get_config(pth)


class Development1H4H(Config):
    """
    Development configuration for 1h_4h environment.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the development configuration for the 1h_4h environment.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_1h_4h"))
        self.configs = Config.get_config(pth)


class Testing(Config):
    """
    Testing configuration.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = True

    def __init__(self, environ):
        """
        Initialize the testing configuration.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_TEST"))
        self.configs = Config.get_config(pth)


class Optimize(Config):
    """
    Optimization configuration.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the optimization configuration.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_OPTIMIZE"))
        self.configs = Config.get_config(pth)


class Docker(Config):
    """
    Docker-specific configuration.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the Docker-specific configuration.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_DOCKER"))
        self.configs = Config.get_config(pth)


class Production(Config):
    """
    Production configuration.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the production configuration.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_PROD"))
        self.configs = Config.get_config(pth)


class Debug(Config):
    """
    Debug configuration.

    Attributes
    ----------
    DEBUG : bool
        Enable or disable debug mode.
    TESTING : bool
        Enable or disable testing mode.
    """

    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        """
        Initialize the debug configuration.

        Parameters
        ----------
        environ : dict
            A dictionary representing the environment variables.
        """
        pth = path.join(basedir, environ.get("CONFIG_PATH_DEBUG"))
        self.configs = Config.get_config(pth)
