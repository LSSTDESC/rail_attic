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

