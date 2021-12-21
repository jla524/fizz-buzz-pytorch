"""
Package wide configurations
Adapted from:
https://github.com/mattcoding4days/kickstart/blob/main/src/kickstart/
"""
from pathlib import Path
from threading import Lock


class ThreadSafeMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    """
    _instances = {}
    _lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=ThreadSafeMeta):
    """
    @description: Global program configuration, uses the dotenv package
     to load runtime configuration from a .env file, once and
     only once into this object, this object can be used through-out
     the code base
    """
    __version = '0.1.0'
    __package: str = __package__
    __base_dir = Path(__file__).resolve(strict=True).parent.parent
    __models_dir = __base_dir / 'models'
    __weights_file = __models_dir / 'weights'

    @classmethod
    def version(cls) -> str:
        """
        @description: getter for the version of package
        """
        return cls.__version

    @classmethod
    def package(cls) -> str:
        """
        @description: getter for the package name
        """
        return cls.__package

    @classmethod
    def base_dir(cls) -> Path:
        """
        @description: getter for base directory
        """
        return cls.__base_dir

    @classmethod
    def models_dir(cls) -> Path:
        """
        @description: getter for the models directory
        """
        return cls.__models_dir

    @classmethod
    def weights_file(cls) -> Path:
        """
        @description: getter for the weights file
        """
        return cls.__weights_file
