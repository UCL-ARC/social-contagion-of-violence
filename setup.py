from setuptools import setup
from src.version import __version__

setup(
   name='social-contagion-of-violence',
   version=__version__,
   description='A Python-based framework to simulate, analyse and infer infections in a network assuming a Hawkes contagion process and accounting for various confounding effects.',
   author='Soumaya Mauthoor',
   author_email='e.lowther@ucl.ac.uk',
   packages=['src'],
   install_requires=['tick', 'networkx', 'mlflow'],
)