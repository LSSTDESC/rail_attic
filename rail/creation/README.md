# RAIL creation modules 

The code here enables the generation of mock photometry corresponding to a fully self-consistent forward model of the joint probability space of redshift and photometry.

## Base design

We will begin with a mock data set of galaxy redshifts and photometry, for example, the DC2 extragalactic catalog.
Because any existing data set is limited in number and coverage in the 6+1 dimensional space of redshifts and photometry, we will expand that mock data set into a continuous probability density in the space of redshifts and photometry.
This process may be done using a GAN to augment the data until it fills out the space, followed by a smoothing or interpolation of that space.
Galaxy redshifts and photometry drawn from that joint probability density will have a true likelihood and a true posterior.
This code may or may not be built off of existing tools made for the testing suite of [`chippr`](https://github.com/aimalz/chippr).

## Future extensions

In the future, we may need to consider a probability space with more data dimensions, such as galaxy images and/or positions in order to consider codes that infer redshifts using photometric information and other sources of information.
Similarly, to evaluate template-fitting codes, we will need to construct the joint probability space of redshifts and photometry from a mock data set of SEDs and redshifts, which could include complex effects like emission lines.

# RAIL degradation modules

The code in the degradation submodule enables the introduction of physical systematics into photometric training/test set pairs via the forward model of the `creation` modules.
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

