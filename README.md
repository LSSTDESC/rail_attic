# RAIL: Redshift Assessment Infrastructure Layers

This repo is home to a series of LSST-DESC projects aiming to quantify the impact of imperfect prior information on probabilistic redshift estimation.
RAIL differs from [PZIncomplete](https://github.com/LSSTDESC/pz_incomplete) in that it is broken into stages, each corresponding to a manageable unit of infrastructure advancement, a specific question, and a potential publication opportunity.
By pursuing the piecemeal development of RAIL, we aim to achieve the broad goals of PZIncomplete.

## Organization

There are four aspects to the RAIL approach, each defined by a minimal version that can be developed further as necessary.
The purpose of each piece of infrastructure is outlined below and described in a README in its own directory, where relevant code will ultimately live.

### creation

To forward-model mock data for testing redshift estimation codes

### degradation

To introduce physical systematics in the mock data sets

### estimation

To automatically execute arbitrary redshift estimation codes

### evaluation

To assess the performance of redshift estimation codes

## Contributing

The RAIL repository uses an issue-branch-review workflow.
When you identify something that should be done, [make an issue](https://github.com/LSSTDESC/RAIL/issues/new) for it.
To contribute, isolate [an issue](https://github.com/LSSTDESC/RAIL/issues) to work on and leave a comment on the issue's discussion page to let others know you're working on it.
Then, make a branch with a name of the form `issue/#/brief-description` and do the work on the branch.
When you're ready to merge your branch into the `master` branch, [make a pull request](https://github.com/LSSTDESC/RAIL/compare) and request that other collaborators review it.
Once the changes have been approved, you can merge and squash the pull request.

## Immediate Plans

