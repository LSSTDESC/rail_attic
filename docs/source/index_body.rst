===============================================
RAIL: Redshift Assessment Infrastructure Layers
===============================================

The LSST-DESC Redshift Assessment Infrastructure Layer (RAIL) code is a framework to perform photometric redshift estimation and analysis for DESC.

The core functionality of RAIL provides tools to ....<continue>

RAIL differs from [PZIncomplete](https://github.com/LSSTDESC/pz_incomplete) in that it is broken into stages, each corresponding to a manageable unit of infrastructure advancement, a specific question, and a potential publication opportunity.
By pursuing the piecemeal development of RAIL, we aim to achieve the broad goals of PZIncomplete.

The source code is publically available at https://github.com/LSSTDESC/RAIL

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/overview
   source/installation
   source/citing

.. toctree::
   :maxdepth: 1
   :caption: Usage Demos

   compiled-demos-examples/basic-creation-demo.rst
   compiled-demos-examples/degradation-demo.rst

.. toctree::
   :maxdepth: 1
   :caption: Reference

   api
