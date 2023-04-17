"""
" Abstract base class defining an Evaluator

The key feature is that the evaluate method.
"""

import numpy as np

from ceci.config import StageParameter as Param
from rail.core.data import Hdf5Handle, QPHandle
from rail.core.stage import RailStage

from rail.evaluation.utils import stat_and_pval
from rail.evaluation.metrics.cdeloss import CDELoss
from qp.metrics.pit import PIT
from rail.evaluation.metrics.pointestimates import PointSigmaIQR, PointBias, PointOutlierRate, PointSigmaMAD


class Evaluator(RailStage):
    """Evaluate the performance of a photo-Z estimator """

    name = 'Evaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(zmin=Param(float, 0., msg="min z for grid"),
                          zmax=Param(float, 3.0, msg="max z for grid"),
                          nzbins=Param(int, 301, msg="# of bins in zgrid"),
                          pit_metrics=Param(str, 'all', msg='PIT-based metrics to include'),
                          point_metrics=Param(str, 'all', msg='Point-estimate metrics to include'),
                          do_cde=Param(bool, True, msg='Evaluate CDE Metric'))
    inputs = [('input', QPHandle),
              ('truth', Hdf5Handle)]
    outputs = [('output', Hdf5Handle)]

    def __init__(self, args, comm=None):
        """Initialize Evaluator"""
        RailStage.__init__(self, args, comm=comm)

    def evaluate(self, data, truth):
        """Evaluate the performance of an estimator

        This will attach the input data and truth to this `Evaluator`
        (for introspection and provenance tracking).

        Then it will call the run() and finalize() methods, which need to
        be implemented by the sub-classes.

        The run() method will need to register the data that it creates to this Estimator
        by using `self.add_data('output', output_data)`.

        Parameters
        ----------
        data : qp.Ensemble
            The sample to evaluate
        truth : Table-like
            Table with the truth information

        Returns
        -------
        output : Table-like
            The evaluation metrics
        """

        self.set_data('input', data)
        self.set_data('truth', truth)
        self.run()
        self.finalize()
        return self.get_handle('output')

    def run(self):
        """ Run method

        Evaluate all the metrics and put them into a table

        Notes
        -----
        Get the input data from the data store under this stages 'input' tag
        Get the truth data from the data store under this stages 'truth' tag
        Puts the data into the data store under this stages 'output' tag
        """

        pz_data = self.get_data('input')
        z_true = self.get_data('truth')['redshift']
        zgrid = np.linspace(self.config.zmin, self.config.zmax, self.config.nzbins+1)

        # Create an instance of the PIT class
        pitobj = PIT(pz_data, z_true)

        # Build reference dictionary of the PIT meta-metrics from this PIT instance
        PIT_METRICS = dict(
            AD=getattr(pitobj, 'evaluate_PIT_anderson_ksamp'),
            CvM=getattr(pitobj, 'evaluate_PIT_CvM'),
            KS=getattr(pitobj, 'evaluate_PIT_KS'),
            OutRate=getattr(pitobj, 'evaluate_PIT_outlier_rate'),
        )

        # Parse the input configuration to determine which meta-metrics should be calculated
        if self.config.pit_metrics == 'all':
            pit_metrics = list(PIT_METRICS.keys())
        else:  #pragma: no cover
            pit_metrics = self.config.pit_metrics.split()

        # Evaluate each of the requested meta-metrics, and store the result in `out_table`
        out_table = {}
        for pit_metric in pit_metrics:
            value = PIT_METRICS[pit_metric]()

            # The result objects of some meta-metrics are bespoke scipy objects with inconsistent fields.
            # Here we do our best to store the relevant fields in `out_table`.
            if isinstance(value, list):  # pragma: no cover
                out_table[f'PIT_{pit_metric}'] = value
            else:
                out_table[f'PIT_{pit_metric}_stat'] = [getattr(value, 'statistic', None)]
                out_table[f'PIT_{pit_metric}_pval'] = [getattr(value, 'p_value', None)]
                out_table[f'PIT_{pit_metric}_significance_level'] = [getattr(value, 'significance_level', None)]

        POINT_METRICS = dict(SimgaIQR=PointSigmaIQR,
                             Bias=PointBias,
                             OutlierRate=PointOutlierRate,
                             SigmaMAD=PointSigmaMAD)
        if self.config.point_metrics == 'all':
            point_metrics = list(POINT_METRICS.keys())
        else:  #pragma: no cover
            point_metrics = self.config.point_metrics.split()

        z_mode = None
        for point_metric in point_metrics:
            if z_mode is None:
                z_mode = np.squeeze(pz_data.mode(grid=zgrid))
            value = POINT_METRICS[point_metric](z_mode, z_true).evaluate()
            out_table[f'POINT_{point_metric}'] = [value]

        if self.config.do_cde:
            value = CDELoss(pz_data, zgrid, z_true).evaluate()
            out_table['CDE_stat'] = [value.statistic]
            out_table['CDE_pval'] = [value.p_value]

        self.add_data('output', out_table)
