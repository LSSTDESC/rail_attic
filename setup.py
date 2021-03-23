from setuptools import setup, find_namespace_packages

packages = find_namespace_packages()
setup(
    name='rail',
    version='0.1.dev0',
    author='The LSST DESC PZWG',
    author_email='aimalz@nyu.edu',
    packages=packages,
    package_data={"": ["*.hdf5", "*.yaml"], "tests": ["*.hdf5", "*.yaml"], },
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
                      'pyyaml',
                      'pandas',
                      'pyarrow',
                      'tables',
                      'astropy',
                      'qp@git+https://github.com/LSSTDESC/qp.git#egg=qp'
                      ],
    extras_require={
        'Full': ['sklearn', 'FlexCode[all]', 'corner', 'dill', 'flax==0.2.0',
                 'jax==0.1.75', 'jaxlib==0.1.52', 'tensorflow==2.2.0',
                 'tensorflow-probability==0.10.1',
                 ],
        'estimation': ['sklearn', 'FlexCode[all]'],
        'creation': ['corner', 'dill', 'flax==0.2.0', 'jax==0.1.75',
                     'jaxlib==0.1.52', 'tensorflow==2.2.0',
                     'tensorflow-probability==0.10.1'],
        'flex': ['FlexCode[all]'],
        'NN': ['sklearn'],
        },
    python_requires='>=3.5',
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
)
