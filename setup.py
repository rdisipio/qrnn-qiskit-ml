#!/usr/bin/env python

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from setuptools import find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = "0.1" # read this from a file such as __init__.py ?

setup(
    name='qrnn-qiskit-ml',
    version=version,
    author="Riccardo Di Sipio <riccardo.disipio@gmail.com>",
    packages=find_packages(),
    url='https://github.com/rdisipio/qrnn-qiskit-ml',
    keywords=[],
    license='LICENSE',
    description='Example of how to use a Recurrent Neural Network with Qiskit ML',
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Programming Language :: Python'
    ],
    scripts=[
        'scripts/run.py',
        'scripts/train.py',
        ],
)
