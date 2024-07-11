
### ~~~
## ~~~ From https://github.com/maet3608/minimal-setup-py/blob/master/setup.py
### ~~~ 

from setuptools import setup, find_packages

setup(
    name = 'quality_of_life',
    version = '2.0.1',
    url = 'https://github.com/ThomasLastName/quality-of-life.git',
    author = 'Thomas Winckelman',
    author_email = 'winckelman@tamu.edu',
    description = 'Helper routines for tasks present in many different projects',
    packages = find_packages(),    
    install_requires = [],
)