**************
Rapid overview
**************

RAIL is the LSST-DESC framework for redshift assessment.
There are three aspects to the RAIL approach: creation, estimation, and evaluation. 
Each is defined by a minimal version that can be developed further as necessary.
The purpose of each piece of infrastructure is outlined below.
RAIL will eventually also comprise a fourth package: summarization.

RAIL differs from `PZIncomplete <https://github.com/LSSTDESC/pz_incomplete>`_ in that it is broken into stages,
each corresponding to a manageable unit of infrastructure advancement, a specific question, and a potential publication opportunity.
By pursuing the piecemeal development of RAIL, we aim to achieve the broad goals of PZIncomplete.

`creation`
==========

Code to forward-model mock data for testing redshift estimation codes, including physical systematics.

`estimation`
============

Code to automatically execute arbitrary redshift estimation codes.

`evaluation`
============

Code to assess the performance of redshift estimation codes.
