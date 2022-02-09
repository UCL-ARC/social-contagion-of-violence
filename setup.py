from setuptools import setup

setup(
   name='social-contagion-of-violence',
   version='1.0',
   description='A Python-based framework to simulate, analyse and infer infections in a network assuming a Hawkes contagion process and accounting for various confounding effects.',
   author='Soumaya Mauthoor',
   author_email='e.lowther@ucl.ac.uk',
   packages=['contagion'],
   install_requires=['tick'],
)