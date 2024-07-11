#
# ~~~ Automatically correspond with the version from setup.py (chat-gpt showed me this one)
from pkg_resources import get_distribution, DistributionNotFound
__version__ = get_distribution('package_name').version