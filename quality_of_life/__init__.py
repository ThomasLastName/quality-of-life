
#
# ~~~ Fetch local package version from setup.py
from pkg_resources import get_distribution, DistributionNotFound
dist = get_distribution(__name__)    # ~~~ tbh I am surprised that __name__ is already defined
__version__ = dist.version

#
# ~~~ Grab some other metadata including the package url from setup.py
__metadata__ = dist.get_metadata(dist.PKG_INFO)
try:
    for line in __metadata__.splitlines():
        if line.startswith('Home-page:'):
            __url__ = line.split(':', 1)[1].strip()
            break
except:
    __url__ = None  # or some default value

#
# ~~~ View the latest version number on the setup.py on github (chat-gpt came up with this one)
def see_latest_version(__url__):
    if __url__ is None:
        raise ValueError("Could not find url of {__name__}")
    #
    # ~~~ Read the contents of the setup.py file
    try:                                                     # ~~~ try to read the setup.py file on github
        import requests, re, os
        username_and_slug = os.path.splitext(__url__[ __url__.find(".com")+len(".com") : ])[0]
        url = "https://raw.githubusercontent.com" + username_and_slug + "/main/setup.py"
        response = requests.get(url)
    except ( requests.ConnectionError, requests.Timeout ):   # ~~~ except don't worry if we don't have an internet connection
        return "no internet"
    #
    # ~~~ Parse the contents of the setup.py file
    if response.status_code == 200:
        setup_py_content = response.text
        version_match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", setup_py_content)
        if version_match:
            latest_version = version_match.group(1)
            return latest_version     # ~~~ a text string
        else:
            raise ValueError("Version string not found in setup.py.")
    else:
        raise ValueError(f"Failed to fetch setup.py: {response.status_code} - {response.text}")

#
# ~~~ Compare the local version number with the version number on the setup.py on github
def check_for_update_available(__version__):
    import warnings
    try:
        latest_version = see_latest_version(__url__)    # ~~~ returns a text sting
        if not latest_version=="no internet":
            if __version__ < latest_version:
                    warnings.warn(
                        f"You are using {dist.project_name} version {__version__}, but version {latest_version} is available. See {__url__} for more information, including how to upgrade.",
                        UserWarning
                    )
                    # print("test")
    except Exception as e:
        warnings.warn(f"Could not check the latest version of {__name__}: {e}")

check_for_update_available(__version__)