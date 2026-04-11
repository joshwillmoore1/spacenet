from . import utils
from . import datasets
from . import partition
from . import pcf

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("spacenet")   
except PackageNotFoundError:
    __version__ = "0.0.0"