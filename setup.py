from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

from src import __version__

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='PIITagging',  # Required

    version=__version__,  # Required
    description='Tagging PII from column names and metadata',
    long_description=long_description,  # Optional

    url='https://github.ford.com/MLSC/PIITagging',
    author='Ford Motor Company',
    author_email='dkrcatov@ford.com',

    packages=find_packages(
        exclude=['docs'])#,
)
