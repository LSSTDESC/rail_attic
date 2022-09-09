# RAIL creation modules 

The code here enables the generation of mock photometry corresponding to a fully self-consistent forward model of the joint probability space of redshift and photometry.

## Base design

The three aspects to RAIL's `creation` subpackage were originally inspired by the testing suite of [`chippr`](https://github.com/aimalz/chippr).

### `Modeler`

A `Modeler` is a stage that yields a forward model of photometry and redshift from which mock data can be drawn.
It may be based on theory, e.g. spectral energy distribution (SED) modeling from stellar population synthesis (SPS), or empirical, based on an existing data set, e.g. the DC2 extragalactic catalog.
The `Modeler` stage effectively informs whatever process it is that can generate a probability space of photometry and redshift.

### `Creator`

A `Creator` is a stage that samples mock data from the probabilistic forward model yielded by a `Modeler`.
Because any existing data set is limited in number and coverage in the 6+1 dimensional space of redshifts and photometry, we expand that mock data set into a continuous probability density in the space of redshifts and photometry.

### `PosteriorCalculator`

Galaxy redshifts and photometry drawn from the joint probability density defined by the `Modeler` will have a true likelihood and a true posterior, which RAIL's `Estimator` algorithms aim to approximate for subsequent comparison.

### Future extensions

In the future, we may need to consider a probability space with more data dimensions, such as galaxy images and/or positions in order to consider codes that infer redshifts using photometric information and other sources of information.
Similarly, to evaluate template-fitting codes, we will need to construct the joint probability space of redshifts and photometry from a mock data set of SEDs and redshifts, which could include complex effects like emission lines.

# RAIL degradation modules

If the aforementioned three aspects of `rail.creation` ensured the self-consistency of mock data, this fourth class ensures realistic complexity via physically motivated forms of systematic error.
Though the high-dimensional probability density outlined in the `creation` directory could be modified in ways that reflect the realistic mismatches between training and test sets, as is implemented in the testing suite for [`chippr`](https://github.com/aimalz/chippr), it is simpler and sufficient (for now) to perform this adjustment in the space of generated mock catalogs from a `Creator`.

## Base design

An initial experimental design would correspond to a single training set and many test sets; 
the systematics that can be implemented under this scheme include imbalances between the training and test sets along the dimensions of brightness, color, and redshift.
Though it is not realistic to think of the universe in this way, realistically complex effects can still be tested in this way.

## Future extensions

An immediate extension could include the projection of measurement errors, such as a bias due to the PSF, aperture photometry parameters, or flux calibration, into the space of photometry.
Some systematics we would like to test, like incorrect values in the training set and blended galaxies, are in essence a form of model misspecification, which may be nontrivial to implement in the space of redshift and photometry probability density and will likely not be possible with a single training set.
All effects will also need to be implemented for SED libraries in order to test template-fitting codes.

