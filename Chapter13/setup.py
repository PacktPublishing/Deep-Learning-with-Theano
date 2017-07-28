# setup.py
from setuptools import setup, find_packages

setup(name='example5',
  version='0.1',
  packages=find_packages(),
  description='keras on gcloud ml-engine',
  license='MIT',
  install_requires=[
      'keras',
      'h5py',
      'nltk'
  ],
  zip_safe=False)
