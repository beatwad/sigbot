import json
from os import path
from dotenv import load_dotenv, find_dotenv
basedir = path.abspath(path.dirname(__file__))
# here we load environment variables from .env, must be called before init. class
load_dotenv(find_dotenv(), verbose=True)


class ConfigFactory(object):
    """ Return configuration settings according to environment variable value """
    @staticmethod
    def factory(environ):
        env = environ.get("ENV", "development")
        if env == 'test':
            return Testing(environ)
        elif env == 'development':
            return Development(environ)
        elif env == '5m_1h':
            return Development5M1H(environ)
        elif env == '1h_1d':
            return Development1H1D(environ)
        elif env == 'docker':
            return Docker(environ)
        elif env == 'production':
            return Production(environ)


class Config:
    """Base config."""
    CONFIG_PATH = ''

    @staticmethod
    def get_config(conf_path):
        with open(conf_path) as f:
            bot_conf = json.load(f)
        return bot_conf


class Development(Config):
    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        pth = path.join(basedir, environ.get('CONFIG_PATH'))
        self.configs = Config.get_config(pth)


class Development5M1H(Config):
    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        pth = path.join(basedir, environ.get('CONFIG_PATH_5m_1h'))
        self.configs = Config.get_config(pth)


class Development1H1D(Config):
    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        pth = path.join(basedir, environ.get('CONFIG_PATH_1h_1d'))
        self.configs = Config.get_config(pth)


class Testing(Config):
    DEBUG = True
    TESTING = True

    def __init__(self, environ):
        pth = path.join(basedir, environ.get('CONFIG_PATH_TEST'))
        self.configs = Config.get_config(pth)


class Docker(Config):
    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        pth = path.join(basedir, environ.get('CONFIG_PATH_DOCKER'))
        self.configs = Config.get_config(pth)


class Production(Config):
    DEBUG = True
    TESTING = False

    def __init__(self, environ):
        pth = path.join(basedir, environ.get('CONFIG_PATH_PROD'))
        self.configs = Config.get_config(pth)
