#from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='rail',
    version='0.1.dev0',
    author='The LSST DESC PZWG',
    author_email='aimalz@nyu.edu',
    packages=find_packages(),
    license='BSD 3-Clause License',
    description="Redshift Assessment Infrastructure Layers",
    long_description=open("README.md").read(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python"
        ],
    install_requires=['numpy'],
    python_requires='>=3.5'
)
