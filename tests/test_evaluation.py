import numpy as np
from rail.evaluation.metrics.pit import PIT, PITOutRate, PITKS, PITCvM, PITAD
from rail.evaluation.metrics.cdeloss import CDELoss
import qp

# values for metrics
OUTRATE = 0.5900099
KSVAL = 0.5964912
CVMVAL = 42.94254
ADVAL_ALL = 224.416922
ADVAL_CUT = -0.655614
CDEVAL = -45.9017221


def construct_test_ensemble():
    np.random.seed(87)
    NPDF = 399
    true_zs = np.random.uniform(high=2.5, size=NPDF)
    locs = np.expand_dims(true_zs + np.random.normal(0.0, 0.01, NPDF), -1)
    scales = np.ones((NPDF, 1)) * 0.1 + np.random.uniform(size=(NPDF, 1)) * .05
    n_ens = qp.Ensemble(qp.stats.norm, data=dict(loc=locs, scale=scales))
    zgrid = np.linspace(0, 1, 301)
    grid_ens = n_ens.convert_to(qp.interp_gen, xvals=zgrid)
    return zgrid, true_zs, grid_ens


def test_pit_metrics():
    zgrid, zspec, pdf_ens = construct_test_ensemble()
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
