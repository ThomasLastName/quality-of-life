#
# ~~~ Automatically correspond with the version from setup.py (chat-gpt showed me this one)
from pkg_resources import get_distribution, DistributionNotFound
__version__ = get_distribution('quality_of_life').version