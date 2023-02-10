#!/usr/bin/env python

from distutils.core import setup

setup(name='lexx',
      version='1.0',
      packages=['src'],
      install_requires=[
          'datasets',
          'transformers',
          'tqdm',
          'torch',
          'numpy',
          'pandas',
      ],
      )
