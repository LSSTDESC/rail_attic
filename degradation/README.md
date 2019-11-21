# RAIL degradation modules

The code here enables the introduction of physical systematics into photometric training/test set pairs via the forward model of the `creation` modules.
(It probably makes more sense for this to be a submodule of the `creation` module.)
The high-dimensional probability density outlined in the `creation` directory can be modified in ways that reflect the realistic mismatches between training and test sets.
Training and test set data will be drawn from such probability spaces with systematics applied in isolation, which preserves the existence of true likelihoods and posteriors.

## Base design

An initial experimental design would correspond to a single training set and many test sets; the systematics that can be implemented under this scheme include imbalances between the training and test sets along the dimensions of brightness, color, and redshift.
Though it is not realistic to think of the universe in this way, realistically complex effects can still be tested in this way.
The "zeroth order" version of this infrastructure could be built using existing tools made for the testing suite of [`chippr`](https://github.com/aimalz/chippr).

## Future extensions

An immediate extension could include the projection of measurement errors, such as a bias due to the PSF, aperture photometry parameters, or flux calibration, into the space of photometry.
Some systematics we would like to test, like incorrect values in the training set and blended galaxies, are in essence a form of model misspecification, which may be nontrivial to implement in the space of redshift and photometry probability density and will likely not be possible with a single training set.
All effects will also need to be implemented for SED libraries in order to test template-fitting codes.

