import os
import numpy as np
from rail.core.stage import RailStage
from rail.core.data import QPHandle, TableHandle
from rail.evaluation.metrics.pit import PIT, PITOutRate, PITKS, PITCvM, PITAD
from rail.evaluation.metrics.cdeloss import CDELoss
import rail.evaluation.metrics.pointestimates as pe
from rail.evaluation.evaluator import Evaluator
import qp


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
    true_ez = (locs.flatten() - true_zs)/(1.+true_zs)
    scales = np.ones((NPDF, 1)) * 0.1 + np.random.uniform(size=(NPDF, 1)) * .05
    n_ens = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))
    zgrid = np.linspace(0, nmax, 301)
    grid_ens = n_ens.convert_to(qp.interp_gen, xvals=zgrid)
    return zgrid, true_zs, grid_ens, true_ez


def test_pit_metrics():
    zgrid, zspec, pdf_ens, _ = construct_test_ensemble()
    pit_obj = PIT(pdf_ens, zspec)
    pit_vals = pit_obj._pit_samps
    quant_grid = np.linspace(0, 1, 101)
    quant_ens, metametrics = pit_obj.evaluate(quant_grid)
    out_rate = PITOutRate(pit_vals, quant_ens).evaluate()
    assert np.isclose(out_rate, OUTRATE)

    ks_obj = PITKS(pit_vals, quant_ens)
    ks_stat = ks_obj.evaluate().statistic
    assert np.isclose(ks_stat, KSVAL)

    cvm_obj = PITCvM(pit_vals, quant_ens)
    cvm_stat = cvm_obj.evaluate().statistic
    assert np.isclose(cvm_stat, CVMVAL)

    ad_obj = PITAD(pit_vals, quant_ens)
    all_ad_stat = ad_obj.evaluate().statistic
    cut_ad_stat = ad_obj.evaluate(pit_min=0.6, pit_max=0.9).statistic
    assert np.isclose(all_ad_stat, ADVAL_ALL)
    assert np.isclose(cut_ad_stat, ADVAL_CUT)

    cde_obj = CDELoss(pdf_ens, zgrid, zspec)
    cde_stat = cde_obj.evaluate().statistic
    assert np.isclose(cde_stat, CDEVAL)


def test_point_metrics():
    zgrid, zspec, pdf_ens, true_ez = construct_test_ensemble()
    zb = pdf_ens.mode(grid=zgrid).flatten()

    ez = pe.PointStatsEz(zb, zspec).evaluate()
    assert np.allclose(ez, true_ez, atol=1.e-2)
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
    pdf = DS.add_data('pdf', pdf_ens, QPHandle)
    truth_table = dict(redshift=true_ez)
    truth = DS.add_data('truth', truth_table, TableHandle)
    evaluator = Evaluator.make_stage(name='Eval')
    evaluator.evaluate(pdf, truth)
    
    os.remove(evaluator.get_output(evaluator.get_aliased_tag('output'), final_name=True))
