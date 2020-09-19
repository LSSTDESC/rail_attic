# from distutils.core import setup
from setuptools import setup, find_namespace_packages

packages = find_namespace_packages()
# packages = find_packages()
# packages.append('rail.estimation.tests.data')
setup(
    name='rail',
    version='0.1.dev0',
    author='The LSST DESC PZWG',
    author_email='aimalz@nyu.edu',
    packages=packages,
    package_data={"": ["*.hdf5", "*.yaml"], "tests":["*.hdf5", "*.yaml"], },
    include_package_data=True,
    license='BSD 3-Clause License',
    description="Redshift Assessment Infrastructure Layers",
    url="https://github.com/LSSTDESC/RAIL",
    long_description=open("README.md").read(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python"
        ],
    install_requires=['numpy',
                      'h5py',
                      'scipy',
                      ],
    python_requires='>=3.5',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
