
### ~~~
## ~~~ From https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~ 

from setuptools import setup, find_packages

setup(
    name = 'quality_of_life',
    version = '2.14.2',
    url = 'https://github.com/ThomasLastName/quality-of-life.git',
    author = 'Thomas Winckelman',
    author_email = 'winckelman@tamu.edu',
    description = 'Helper routines for tasks present in many different projects',
    packages = find_packages(),
    install_requires = [
        "numpy",
        "scipy",
        "matplotlib",
        "plotly",
        "tqdm",
        "requests"
    ],
)