# RAIL creation modules

The code here enables the generation of mock photometry scorresponding to a fully self-consistent model of the joint probability space of redshift and photometry.

## Base design

We will begin with a mock data set of galaxy redshifts and photometry.
Because any existing data set is sparse in this space and smaller than the number of galaxies we might want to test redshift estimation codes, we will develop that mock data set into a continuous probability density in the space of redshifts and photometry.
This process may be done using a GAN to augment the data until it fills out the space, followed by a smoothing or interpolation of that space.

The high-dimensional probability density can be modified in ways that reflect the realistic mismatches between training and test sets, a process executed in the `degradation` modules.
Training and test set data will be drawn from such probability spaces, meaning each galaxy has a true likelihood and a true posterior with which to compare.
This code may or may not be built off of tools made for the testing suite of [`chippr`](https://github.com/aimalz/chippr).

## Future extensions

In the future, we may need to integrate this with galaxy positions in order to consider redshift estimation codes that jointly use position and photometric information.
Similarly, to evaluate template-fitting codes, we will need to construct the joint probability space of redshifts and photometry from a mock data set of SEDs and redshifts, which could include complex effects like emission lines.

