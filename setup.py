from setuptools import setup, find_namespace_packages

# basic dependencies for all RAIL modules
install_requires = [
    "astropy",
    "h5py",
    "numpy",
    "pandas>=1.1",
    "tables-io",
    "qp @ git+https://github.com/LSSTDESC/qp",
]

# dependencies for the Creation module
creation_extras = ["pzflow"]

# dependencies required for all estimators in the Estimation module
estimation_extras = [
    "matplotlib",
    "pyarrow",
    "pyyaml",
    "scipy>=1.5.2",
    "tables",
]
# dependencies for specific estimators in the Estimation module
estimation_codes = {
    "bpz": ["DESC_BPZ @ git+https://github.com/LSSTDESC/DESC_BPZ"],
    "flex": ["FlexCode[all]"],
    "NN": ["sklearn"],
}

# dependencies for the Evaluation module
evaluation_extras = []


# compile the extras_require dictionary
extras_require = dict()
extras_require["creation"] = creation_extras
extras_require["estimation"] = estimation_extras + list(
    set(sum(estimation_codes.values(), []))
)
for key, values in estimation_codes.items():
    extras_require[key] = estimation_extras + values
extras_require["evaluation"] = evaluation_extras
extras_require["all"] = extras_require["full"] = extras_require["Full"] = list(
    set(
        (
            extras_require["creation"]
            + extras_require["estimation"]
            + extras_require["evaluation"]
        )
    )
)

# load the version number
with open("rail/version.py") as f:
    __version__ = f.read().replace('"', "").split("=")[1]

# setup the rail package!
setup(
    name="rail",
    version=__version__,
    author="The LSST DESC PZ WG",
    author_email="aimalz@nyu.edu",
    packages=find_namespace_packages(),
    package_dir={'rail': './rail','rail.estimation':'./rail/estimation','rail.estimation.algos': './rail/estimation/algos','rail.estimation.algos.include_delightPZ':'./rail/estimation/algos/include_delightPZ'},
    package_data={
        "": ["*.hdf5", "*.yaml", "*.sed", "*.res", "*.AB", "*.list", "*.columns"],
        "tests": ["*.hdf5", "*.yaml", "*.columns"],
        "rail/estimation/data/SED": ["*.sed", "*.list"],
        "rail/estimation/data/FILTER": ["*.res"],
        "rail/estimation/data/AB": ["*.AB"],
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
        "Programming Language :: Python",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.5",
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
