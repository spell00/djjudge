import sys

from distutils.core import setup
from setuptools import find_packages


CURRENT_PYTHON = sys.version_info[:2]

setup(name="djjudge", version="0.2",
      description="A package for learning to classify raw audio according to a user's self-definied "
                  "scores of appreciation",
      url="https://github.com/pytorch/audio",
      packages=find_packages(),
      author="Simon J Pelletier",
      author_email="simonjpelletier@gmail.com",
      install_requires=['torch',
                        'matplotlib',
                        'tensorflow',
                        'numpy',
                        'inflect',
                        'librosa',
                        'scipy',
                        'tensorboardX',
                        'Unidecode',
                        'magenta',
                        'apex'
                        ]
      )
