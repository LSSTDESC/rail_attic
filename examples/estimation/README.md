## Using the `main.py` example script

As a quick way to get started running a photo-z estimator, we have provided the `main.py` script that can be run on the command line with:
`python main.py [configfile] [basefile]`
where `configfile` is a yaml file containing the parameters specific to an individual algorithm, and `basefile` is a separate yaml file that controls some top-level parameters.  If `basefile` is not specified, the code will run using the parameters in `base.yaml` in this directory.

An example set of configuration parameters with sensible defaults for each code is located in the `configs` subdirectory, and each should run with the default base.yaml that points to a training set of 10,000 galaxies and validation set of 20,000 galaxies located in the `RAIL/tests/data/` directory that ships with the code.  So, for example, to run BPZ_lite on these default datasets, you should be able to simply run:
`python main.py configs/BPZ_lite.yaml`

`main.py` writes result files to `results/[codename]` on completion.

The configuration parameters differ code-by-code, you can see a brief description of each in `RAIL/estimation/algos/README.md`, or the description string in the default dictionary in each estimator's specific subclass.

the parameters in base.yaml are as follows, all of which are *required* for the code to run properly, and should be stored in a base dictionary called `base_config`:

- `trainfile`: str, specifies the path to the file containing training data

- `testfile`: str, specifies the path to the file of test data

- `hdf5_groupname`: str, name of any top-level hdf5 group in which the training and test data resides, e.g. `photometry` for our example data.  If the data groupname is not necessary, yaml expects this to be `null` (not None as in python)

- `chunk_size`: int, if your dataset is large the code processes the test file in chunks to minimize memory usage, `chunk_size` specifies how many galaxies to process in each data chunk.

- `config_path`: str, location of config files, not actually used currently

- `outpath`: str, location to write the results to, results will be written to a subdirectory of `outpath` with the name of the particular photo-z subclass.

- `output_format`: str, this controls whether main.py writes either an ordered dictionary or a qp Ensemble for the output PDFs.  specifying "qp" will return an Ensemble, specifying "old" (or anything other than "qp", really), will return an ordered dictionary.
