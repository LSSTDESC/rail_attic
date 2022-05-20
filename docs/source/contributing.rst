************
Contributing
************

The RAIL repository uses an issue-branch-review workflow.
When you identify something that should be done, `make an issue <https://github.com/LSSTDESC/RAIL/issues/new>`_
for it.   
We ask that if applicable and you are comfortable doing so, you add labels to the issue to
mark what part of the code base it relates to, its priority level, and if it's well-suited to newcomers, as opposed to requiring more familiarity with the code or technical expertise.   
Also, if you intend
to work on the issue yourself, please assign the issue to yourself.

To contribute, isolate `an issue <https://github.com/LSSTDESC/RAIL/issues>`_ to work on, assign yourself, and leave a comment on
the issue's discussion page to let others know you're working on it. 
Then, make a branch with a name of the
form `issue/[#]/brief-description` and make changes in your branch. 
While developing in a branch, don't forget to pull from `main` regularly to make sure your work is compatible with other recent changes.

Before you make a pull request we ask that you do two things:
   1. Run `pylint` and clean up the code accordingly.  You may need to
      install `pylint` to do this.
   2. Add unit tests and make sure that the new code is fully
      `covered`.   You make need to install `pytest` and `pytest-cov`
      to do this.  You can use the `do_cover.sh` script in the top
      level directory to run `pytest` and generate a coverage report.

When you're ready to merge your branch into the `main` branch,
`make a pull request <https://github.com/LSSTDESC/RAIL/compare>`_, and request that other team members review it if you have any in mind, for example, those who have consulted on some of the work.
Once the changes have been approved, you can merge and squash the pull request as well as close its corresponding issue by putting `closes #[#]` in the comment closing the pull request.

To review a pull request, it's a good idea to start by pulling the changes and running the unit tests (see above). If there are no problems with that, you can make suggestions for optional improvements (e.g. adding a one-line comment before a clever block of code or including a demonstration of new functionality in the example notebooks) or request necessary changes (e.g. including an exception for an edge case that will break the code or separating out code that's repeated in multiple places).
