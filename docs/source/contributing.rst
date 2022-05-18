************
Contributing
************

The RAIL repository uses an issue-branch-review workflow.
When you identify something that should be done, `make an issue <https://github.com/LSSTDESC/RAIL/issues/new>`_
for it.   We ask that if applicable and you are comfortable doing so, you add labels to the issue to
mark what part of the code base it relates to.   Also, if you intend
to work on the issue yourself, please assign the issue to yourself.

To contribute, isolate `an issue <https://github.com/LSSTDESC/RAIL/issues>`_ to work on and leave a comment on
the issue's discussion page to let others know you're working on it. Then, make a branch with a name of the
form `issue/#/brief-description` and do the work on the branch.

Before you make a pull request we ask that you do two things:
   1. Run `pylint` and clean up the code accordingly.  You may need to
      install `pylint` to do this.
   2. Add unit tests and make sure that the new code is fully
      `covered`.   You make need to install `pytest` and `pytest-cov`
      to do this.  You can use the `do_cover.sh` script in the top
      level directory to run `pytest` and generate a coverage report.

When you're ready to merge your branch into the `main` branch,
`make a pull request <https://github.com/LSSTDESC/RAIL/compare>`_ and request that other collaborators review it.
Once the changes have been approved, you can merge and squash the pull request.

