from setuptools import find_namespace_packages, setup

# basic dependencies for all RAIL modules
install_requires = [
    "h5py",
    "numpy",
    "pandas>=1.1",
    "tables-io>=0.7.5",
    "ceci>=1.10.1",
    "qp @ git+https://github.com/LSSTDESC/qp",
    "pyyaml",
    "scipy>=1.9.0",
]

# dependencies for the core module
core_extras = ["hyperbolic @ git+https://github.com/jlvdb/hyperbolic"]

# dependencies for the Creation module
creation_extras = ["pzflow"]

# dependencies required for all estimators in the Estimation module
estimation_extras = []

# dependencies for specific estimators in the Estimation module
estimation_codes = {
    "NN": ["sklearn"],
}

# dependencies for the Evaluation module
evaluation_extras = []


# compile the extras_require dictionary
extras_require = dict()
extras_require["core"] = core_extras
extras_require["creation"] = creation_extras
extras_require["estimation"] = estimation_extras + list(
    set(sum(estimation_codes.values(), []))
)
for key, values in estimation_codes.items():
    extras_require[key] = estimation_extras + values
extras_require["evaluation"] = evaluation_extras
extras_require["base"] = list(
    set(
        (
            extras_require["core"]
            + extras_require["creation"]
            + extras_require["estimation"]
            + extras_require["evaluation"]
        )
    )
)

extras_require["all"] = extras_require["full"] = extras_require["Full"] = extras_require["base"]


# setup the rail package!
setup(
    name="pz-rail",
    author="The LSST DESC PZ WG",
    author_email="aimalz@nyu.edu",
    packages=find_namespace_packages(),
    package_dir={
        "rail": "./rail",
        "rail.estimation": "./rail/estimation",
        "rail.estimation.algos": "./rail/estimation/algos",
    },
    package_data={
        "": [
            "*.hdf5",
            "*.yaml",
            "*.sed",
            "*.res",
            "*.AB",
            "*.list",
            "*.columns",
            "*.pkl",
        ],
        "tests": ["*.hdf5", "*.yaml", "*.columns"],
        "examples/estimation/data/SED": ["*.list"],
        "examples/estimation/data/AB": ["*.AB"],
        "examples/estimation/data/FILTER": ["*.res", "*.txt"],
        "examples/goldenspike/data": ["*.pkl"],
    },
    include_package_data=True,
    license="BSD 3-Clause License",
    description="Redshift Assessment Infrastructure Layers",
    url="https://github.com/LSSTDESC/RAIL",
    long_description=open("README.md").read(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
    setup_requires=["setuptools_scm", "pytest-runner"],
    use_scm_version={"write_to": "rail/core/_version.py"},
    tests_require=["pytest"],
)
