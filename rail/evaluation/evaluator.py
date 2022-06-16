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
from rail.evaluation.metrics.pit import PIT, PITOutRate, PITKS, PITCvM
from rail.evaluation.metrics.pointestimates import PointSigmaIQR, PointBias, PointOutlierRate, PointSigmaMAD


class Evaluator(RailStage):
    """Evalute the preformance of a photo-Z estimator """

    name = 'Evaluator'
    config_options = RailStage.config_options.copy()
    config_options.update(zmin=Param(float, 0., msg="min z for grid"),
                          zmax=Param(float, 3.0, msg="max z for grid"),
                          nzbins=Param(int, 301, msg="# of bins in zgrid"),
                          pit_metrics=Param(str, 'all', msg='PIT-based metrics to include'),
                          point_metrics=Param(str, 'all', msg='Point-estimate metrics to include'),
                          do_cde=Param(bool, True, msg='Evalute CDE Metric'))
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

        PIT_METRICS = dict(KS=PITKS,
                           CvM=PITCvM,
                           OutRate=PITOutRate)
        if self.config.pit_metrics == 'all':
            pit_metrics = list(PIT_METRICS.keys())
        else:  #pragma: no cover
            pit_metrics = self.config.pit_metrics.split()

        POINT_METRICS = dict(SimgaIQR=PointSigmaIQR,
                             Bias=PointBias,
                             OutlierRate=PointOutlierRate,
                             SigmaMAD=PointSigmaMAD)
        if self.config.point_metrics == 'all':
            point_metrics = list(POINT_METRICS.keys())
        else:  #pragma: no cover
            point_metrics = self.config.point_metrics.split()

        out_table = {}
        pitobj = None
        for pit_metric in pit_metrics:
            if pitobj is None:
                pitobj = PIT(pz_data, z_true)
                quant_ens, _ = pitobj.evaluate()
                pit_vals = np.array(pitobj.pit_samps)

            value = PIT_METRICS[pit_metric](pit_vals, quant_ens).evaluate()
            if isinstance(value, stat_and_pval):
                out_table[f'PIT_{pit_metric}_stat'] = [value.statistic]
                out_table[f'PIT_{pit_metric}_pval'] = [value.p_value]
            elif isinstance(value, float):
                out_table[f'PIT_{pit_metric}'] = [value]

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
