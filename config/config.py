import json
import os
from os import environ, path
from dotenv import load_dotenv, find_dotenv

environ["ENV"] = "development"
print(environ)
basedir = path.abspath(path.dirname(__file__))
# here we load environment variables from .env, must be called before init. class
load_dotenv(find_dotenv(), verbose=True)


class ConfigFactory(object):
    @staticmethod
    def factory():
        env = environ.get("ENV", "development")
        if env == 'testing':
            return Testing()
        elif env == 'development':
            return Development()
        elif env == 'docker':
            return Docker()
        elif env == 'production':
            return Production()


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

    def __init__(self):
        pth = path.join(basedir, environ.get('CONFIG_PATH'))
        self.configs = Config.get_config(pth)


class Testing(Config):
    DEBUG = True
    TESTING = True

    def __init__(self):
        pth = path.join(basedir, environ.get('CONFIG_PATH_TEST'))
        self.configs = Config.get_config(pth)


class Docker(Config):
    DEBUG = True
    TESTING = False

    def __init__(self):
        pth = path.join(basedir, environ.get('CONFIG_PATH_DOCKER'))
        self.configs = Config.get_config(pth)


class Production(Config):
    DEBUG = True
    TESTING = False

    def __init__(self):
        pth = path.join(basedir, environ.get('CONFIG_PATH_PROD'))
        self.configs = Config.get_config(pth)


if __name__ == '__main__':
    cf = ConfigFactory().factory().configs
    print(cf)
