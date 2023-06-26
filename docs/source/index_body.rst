===============================================
RAIL: Redshift Assessment Infrastructure Layers
===============================================

RAIL is a flexible software library providing tools to produce at-scale photometric redshift data products, including uncertainties and summary statistics, and stress-test them under realistically complex systematics.

RAIL serves as the infrastructure supporting many extragalactic applications of `the Legacy Survey of Space and Time (LSST) <https://www.lsst.org/>`_ on `the Vera C. Rubin Observatory<https://rubinobservatory.org/>`_, including Rubin-wide commissioning activities. 
RAIL was initiated by the Photometric Redshifts (PZ) Working Group (WG) of the `LSST Dark Energy Science Collaboration (DESC) <https://lsstdesc.org/>`_ as a result of the lessons learned from the `Data Challenge 1 (DC1) experiment <https://academic.oup.com/mnras/article/499/2/1587/5905416>`_ to enable the PZ WG Deliverables in the `LSST-DESC Science Roadmap (see Sec. 5.18) <https://lsstdesc.org/assets/pdf/docs/DESC_SRM_latest.pdf>`_, aiming to guide the selection and implementation of redshift estimators in DESC analysis pipelines.

RAIL is developed and maintained by a diverse team comprising DESC Pipeline Scientists (PSs), international in-kind contributors, LSST Interdisciplinary Collaboration for Computing (LINCC) Frameworks software engineers, and other volunteers, but all are welcome to join the team regardless of LSST data rights. 
To get involved, chime in on the issues in any of the RAIL repositories described in the Overview section.


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   source/installation
   source/citing
   source/overview
   source/contributing
   
.. toctree::
   :maxdepth: 1
   :caption: Usage Demos

   source/core-notebooks
   source/creation-notebooks
   source/estimation-notebooks
   source/other-notebooks
