# RAIL degradation modules

The code here enables the introduction of physical systematics into photometric training/test set pairs via the forward model of the `creation` modules.
(It probably makes more sense for this to be a submodule of the `creation` module.)

## Base design

An initial experimental design would correspond to a single training set and many test sets; the systematics that can be implemented under this scheme include imbalances between the training and test sets.

## Future extensions

Some systematics we would like to test, like incorrect values in the training set and blended galaxies, will require model misspecification, which may be nontrivial to implement in the space of redshift and photometry probability density.
These effects will also need to be implemented for SED libraries in order to test template-fitting codes.

