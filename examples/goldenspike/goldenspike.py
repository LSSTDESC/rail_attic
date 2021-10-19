import astropy.table
import numpy as np
import yaml
import matplotlib.pyplot as plt
from pzflow import Flow
from rail.creation import Creator
import rail.creation.degradation
from rail.estimation.estimator import Estimator
from rail.evaluation.metrics.pit import *
import rail.evaluation.metrics.pointestimates as pe
from rail.evaluation.metrics.cdeloss import CDELoss
import tables_io
import qp
import spike_utils
from eval_utils import ks_plot, plot_pit_qq, plot_point_est


def main():
    with open("creation_goldspike.yaml", 'r') as f:
        c_par = yaml.safe_load(f)
    with open("estimation_goldspike.yaml", 'r') as f:
        est_par = yaml.safe_load(f)
    with open("evaluation_goldspike.yaml", 'r') as f:
        eval_par = yaml.safe_load(f)

    ####CREATION
    if c_par['has_flow']:
        flow = Flow(file=c_par['flow_file'])
    else:
        if c_par['use_local_data']:
            datafile = c_par['local_flow_data_file']
            print(f"constructing flow from file {datafile}")
            raw_data = tables_io.io.read(datafile, tType=tables_io.types.PD_DATAFRAME)
            # trim data to only columns wanted for pzflow training
            flow_columns = c_par['flow_columns']
            data = raw_data[flow_columns]
            flow = spike_utils.generateflow(data)
        if c_par['save_flow']:
            flow.save(c_par['saved_flow_file'])

    # creator for Test data
    creator = Creator(flow)

    if c_par['use_degrader']:
        try:
            deg_mod = getattr(rail.creation.degradation, c_par['degrader_name'])
            degrader = deg_mod(**c_par['degrader_args'])
            print(f"using degrader {c_par['degrader_name']}")
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f" module {c_par['degrader_name']} not found!")
        degraded_creator = Creator(flow, degrader=degrader)
    else:
        degraded_creator = creator

    # create samples for both test and train
    # if use_degrader is False then data is generated from the same ensemble
    test_data = creator.sample(c_par['N_test_gals'])
    z_true = test_data['redshift']
    train_data = degraded_creator.sample(c_par['N_train_gals'])

    # create redshift posteriors for each of the sample galaxies in the test sample
    zgrid = np.linspace(c_par['zmin'], c_par['zmax'], c_par['nzbins'])
    test_pdfs = flow.posterior(test_data, column=c_par['z_column'], grid=zgrid)
    test_ens = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=test_pdfs))
    # write out test_posteriors to file
    if c_par['save_ensemble']:
        test_ens.write_to(c_par['ensemble_file'])

    print("created test data and ensembles, calculating magnitude errors...")

    # We need to add mag errors to our data, we can use the PhotoZDC1 package
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    for dset in [test_data, train_data]:
        for band in bands:
            tmperr = spike_utils.make_errors(dset[f'mag_{band}_lsst'],
                                             f'LSST{band}')
            dset[f'mag_err_{band}_lsst'] = tmperr

    # SAVE DATA TO FILE:
    test_data.to_parquet(c_par['test_filename'])
    train_data.to_parquet(c_par['train_filename'])
    # Prepare data to be used in estimation. Creation spits out Pandas dataframe
    # convert to dictionary of arrays.
    input_test = tables_io.convert(test_data, tables_io.types.NUMPY_DICT)
    input_train = tables_io.convert(train_data, tables_io.types.NUMPY_DICT)

    #####Estimation
    print("starting estimation...")

    # iterate through list of codes in config
    base_yaml = est_par['base_yaml']
    res_file = est_par['results_file']
    result_base = est_par['estimation_results_base']
    estimators = est_par['estimators']
    print("list of estimators:")
    print(estimators)
    table = "Code        | PIT outrate |     KS      |     CvM     |  CDE loss   |    sigma    |    bias     | fout \n"
    for est_key in estimators:
        print(f"running estimator {est_key}")
        est_dict = estimators[est_key]

        name = est_dict['run_params']['class_name']
        try:
            Estimator._find_subclass(name)
        except KeyError:
            raise ValueError(f"Class name {name} for PZ code is not defined")

        code = Estimator._find_subclass(name)
        pz = code(base_yaml, est_dict)

        pz.inform_dict = est_dict['run_params']['inform_options']
        if pz.inform_dict['load_model']:
            # note: specific options set in subclasss func def
            pz.load_pretrained_model()
        else:
            pz.inform(input_train) # train on training data

        # don't bother with iterator for now, just run the entire chunk
        pz_data = pz.estimate(input_test)

        #save data?
        if est_par['save_pdfs']:
            outfile = result_base + f"{name}.pq"
            pz_data.write_to(outfile)

        print(f"finished estimation using code {name}")

        # ####Evaluation
        # calculate PIT values, takes a qp ensemble and the true z's
        pitobj = PIT(pz_data, z_true)
        quant_ens, metamets = pitobj.evaluate()
        pit_vals = np.array(pitobj._pit_samps)
        # KS
        ksobj = PITKS(pit_vals, quant_ens)
        ks_stat_and_pval = ksobj.evaluate()
        print(f"KS value for code {name}: {ks_stat_and_pval.statistic}")
        cvmobj = PITCvM(pit_vals, quant_ens)
        cvm_stat_and_pval = cvmobj.evaluate()
        print(f"CvM value for code {name}: {cvm_stat_and_pval.statistic}")
        cdelossobj = CDELoss(pz_data, zgrid, z_true)
        cde_stat_and_pval = cdelossobj.evaluate()
        print(f"CDE Loss for code {name}: {cde_stat_and_pval.statistic}")

        if eval_par['make_ks_figure']:
            figdir = eval_par['figure_directory']
            tmpname = f"{figdir}/{name}_ksplot.jpg"
            ks_plot(pitobj, 101, True, tmpname)

        pit_out_rate = PITOutRate(pit_vals, quant_ens).evaluate()
        print(f"PIT Outlier Rate for code {name}: {pit_out_rate}")

        z_mode = pz_data.mode(grid=zgrid)
        sigma_iqr = pe.PointSigmaIQR(z_mode, z_true).evaluate()
        bias = pe.PointBias(z_mode, z_true).evaluate()
        frac = pe.PointOutlierRate(z_mode, z_true).evaluate()
        sigma_mad = pe.PointSigmaMAD(z_mode, z_true).evaluate()
        print(f"sigma_IQR for {name}: {sigma_iqr}\nbias for {name}: {bias}")
        print(f"cat. outlier frac for {name}: {frac}")

        table += f"{name:11s} | {pit_out_rate:11.4f} | {ks_stat_and_pval.statistic:11.4f}"
        table += f" | {cvm_stat_and_pval.statistic:11.4f} | {cde_stat_and_pval.statistic:11.4f}"
        table += f" | {sigma_iqr:11.4f} | {bias:11.4f} | {frac:11.4f}\n"
        
        plot_point_est(z_mode, z_true, sigma_iqr, name,
                       f"{figdir}/{name}_pointests.jpg")

        # pdfdata = pz_data.objdata()['yvals']
        # xzgrid = pz_data.metadata()['xvals']

        plot_pit_qq(pz_data, z_true, qbins=101, title="PIT-QQ",
                    code=f"{name}", pit_out_rate=pit_out_rate, outdir='results',
                    savefig=True)

    with open(res_file, "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
