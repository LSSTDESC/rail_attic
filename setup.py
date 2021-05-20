from setuptools import setup, find_namespace_packages

packages = find_namespace_packages()
setup(
    name="rail",
    version="0.1.dev0",
    author="The LSST DESC PZWG",
    author_email="aimalz@nyu.edu",
    packages=packages,
    package_data={
        "": ["*.hdf5", "*.yaml"],
        "tests": ["*.hdf5", "*.yaml"],
    },
    include_package_data=True,
    license="BSD 3-Clause License",
    description="Redshift Assessment Infrastructure Layers",
    url="https://github.com/LSSTDESC/RAIL",
    long_description=open("README.md").read(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
<<<<<<< HEAD
        "Programming Language :: Python",
    ],
    install_requires=[
        "h5py",
        "numpy",
        "scipy",
        "pyyaml",
        "pandas>=1.1",
        "pyarrow",
        "tables",
    ],
=======
        "Programming Language :: Python"
        ],
    install_requires=['numpy',
                      'h5py',
                      'scipy>=1.6.0',
                      'pyyaml',
                      'pandas',
                      'pyarrow',
                      'tables',
                      ],
>>>>>>> 81ff354dc1a353a86c65be26a30bc5eca1929b11
    extras_require={
        "Full": ["sklearn", "FlexCode[all]", "pzflow"],
        "estimation": ["sklearn", "FlexCode[all]"],
        "creation": ["pzflow"],
        "flex": ["FlexCode[all]"],
        "NN": ["sklearn"],
    },
    python_requires=">=3.5",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
