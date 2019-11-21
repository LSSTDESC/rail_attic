# RAIL estimation modules

This code enables the automatic execution of arbitrary redshift estimation codes in a common computing environment.

## Base design

To run a challenge, we will require scripts that accept test set photometry, run a particular pre-trained photo-z estimation code, and produce photo-z posteriors p(z | photometry).
Contributors will thus be given a single training set and will provide to us a pre-trained code wrapped by a script built according to the templates found in this directory.

## Future extensions

It may not be possible to isolate some complex `degradation` effects in a shared training set, so future versions will require an additional script for each machine-learning-based code that 
For codes that do not naively apply machine learning to photometry, we will need to provide information beyond photometry; this would enable the inclusion of codes that jointly use position information and those requiring SED libraries.
Decisions need to be made about the output format of redshift posteriors.
