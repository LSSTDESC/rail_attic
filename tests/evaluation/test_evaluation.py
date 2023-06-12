import os

import numpy as np
import qp

import rail.evaluation.metrics.pointestimates as pe
from rail.core.data import QPHandle, TableHandle
from rail.core.stage import RailStage
from rail.evaluation.evaluator import Evaluator
from rail.evaluation.metrics.cdeloss import CDELoss

# values for metrics
OUTRATE = 0.0
KSVAL = 0.367384
CVMVAL = 20.63155
ADVAL_ALL = 82.51480
ADVAL_CUT = 1.10750
CDEVAL = -4.31200
SIGIQR = 0.0045947
BIAS = -0.00001576
OUTRATE = 0.0
SIGMAD = 0.0046489


def construct_test_ensemble():
    np.random.seed(87)
    nmax = 2.5
    NPDF = 399
    true_zs = np.random.uniform(high=nmax, size=NPDF)
    locs = np.expand_dims(true_zs + np.random.normal(0.0, 0.01, NPDF), -1)
    true_ez = (locs.flatten() - true_zs) / (1.0 + true_zs)
    scales = np.ones((NPDF, 1)) * 0.1 + np.random.uniform(size=(NPDF, 1)) * 0.05
    n_ens = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))
    zgrid = np.linspace(0, nmax, 301)
    grid_ens = n_ens.convert_to(qp.interp_gen, xvals=zgrid)
    return zgrid, true_zs, grid_ens, true_ez


def test_point_metrics():
    zgrid, zspec, pdf_ens, true_ez = construct_test_ensemble()
    zb = pdf_ens.mode(grid=zgrid).flatten()

    ez = pe.PointStatsEz(zb, zspec).evaluate()
    assert np.allclose(ez, true_ez, atol=1.0e-2)
    # grid limits ez vals to ~10^-2 tol

    sig_iqr = pe.PointSigmaIQR(zb, zspec).evaluate()
    assert np.isclose(sig_iqr, SIGIQR)

    bias = pe.PointBias(zb, zspec).evaluate()
    assert np.isclose(bias, BIAS)

    out_rate = pe.PointOutlierRate(zb, zspec).evaluate()
    assert np.isclose(out_rate, OUTRATE)

    sig_mad = pe.PointSigmaMAD(zb, zspec).evaluate()
    assert np.isclose(sig_mad, SIGMAD)


def test_evaluation_stage():
    DS = RailStage.data_store
    zgrid, zspec, pdf_ens, true_ez = construct_test_ensemble()
    pdf = DS.add_data("pdf", pdf_ens, QPHandle)
    truth_table = dict(redshift=zspec)
    truth = DS.add_data("truth", truth_table, TableHandle)
    evaluator = Evaluator.make_stage(name="Eval")
    evaluator.evaluate(pdf, truth)

    os.remove(evaluator.get_output(evaluator.get_aliased_tag("output"), final_name=True))
