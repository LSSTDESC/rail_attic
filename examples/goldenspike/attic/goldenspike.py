import os
import numpy as np
import yaml
import pandas as pd
from pzflow import Flow
from rail.creation import Creator
import rail.creation.degradation
from rail.creation.degradation.lsst_error_model import LSSTErrorModel
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

    # ###CREATION
    if c_par['run_creation']:
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

        # we need magnitude errors for both the test and train data, so we will
        # make an LSSTErrorModel degrader to apply in our truth Creator:
        err_params = c_par['LSSTErrorModel_params']
        err_degrader = LSSTErrorModel(**err_params)
        # calculate one sigma limiting mags to use later for non-detections
        onesig_lims = err_degrader.get_limiting_mags(Nsigma=1, coadded=True)

        # creator for Test data
        creator = Creator(flow, degrader=err_degrader)

        # create samples for both test and train
        # if use_degrader is False then data is generated from the same ensemble
        test_data = creator.sample(c_par['N_test_gals'])
        z_true = test_data['redshift']
        train_data = creator.sample(c_par['N_train_gals'])

        if c_par['use_degraders']:
            seed = c_par['degrader_seed']
            degraders = c_par['degraders']
            for degr in degraders:
                try:
                    degrader = getattr(rail.creation.degradation, degr)
                    deg_dict = degraders[degr]
                    print(f"using degrader {degr}")
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(f" module {degr} not found!")
                train_data = degrader(**deg_dict)(train_data, seed)

        # KLUDGE: in estimation we've been using e.g. mag_u_lsst and mag_err_u_lsst
        # Will rename the columns by hand to test
        rename_dict = c_par['rename_cols']
        train_data.rename(columns=rename_dict, inplace=True)
        test_data.rename(columns=rename_dict, inplace=True)

        # ANOTHER KLUDGE: this time to put 1 sigma error in mag_err column
        # for non-detections, as expected by BPZ, FZBoost.
        for data in [train_data, test_data]:
            for band in ['u', 'g', 'r', 'i', 'z', 'y']:
                mask = (np.isclose(data[f'mag_{band}_lsst'], 99.0))
                data[f'mag_err_{band}_lsst'][mask] = onesig_lims[f'mag_{band}_lsst']

        # create redshift posteriors for each of the sample galaxies in the test sample
        zgrid = np.linspace(c_par['zmin'], c_par['zmax'], c_par['nzbins'])
        test_pdfs = flow.posterior(test_data, column=c_par['z_column'], grid=zgrid)
        test_ens = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=test_pdfs))
        # write out test_posteriors to file
        if c_par['save_ensemble']:
            test_ens.write_to(c_par['ensemble_file'])

        # SAVE DATA TO FILE:
        if not os.path.exists(c_par['saved_data_dir']):
            print(f"directory {c_par['saved_data_dir']} doesn't exist, creating...")
            os.makedirs(c_par['saved_data_dir'])
        # convert to ordered dict for either write to file or use by Estimation
        input_test = tables_io.convert(test_data, tables_io.types.NUMPY_DICT)
        input_train = tables_io.convert(train_data, tables_io.types.NUMPY_DICT)
        tables_io.io.write(input_test,
                           os.path.join(c_par['saved_data_dir'],
                                        c_par['test_filename']), fmt='hdf5')
        tables_io.io.write(input_train,
                           os.path.join(c_par['saved_data_dir'],
                                        c_par['train_filename']), fmt='hdf5')

    # ####Estimation

    # iterate through list of codes in config
    base_yaml = est_par['base_yaml']
    with open(base_yaml, "r") as f:
        base_params = yaml.safe_load(f)

    if not c_par['run_creation']:
        # load training and test data from files specified in previous run of creation
        testpath = os.path.join(c_par['saved_data_dir'], c_par['test_filename'])
        input_test = tables_io.io.read(testpath+".hdf5")
        z_true = input_test['redshift']
        zgrid = np.linspace(c_par['zmin'], c_par['zmax'], c_par['nzbins'])
        trainpath = os.path.join(c_par['saved_data_dir'], c_par['train_filename'])
        input_train = tables_io.io.read(trainpath+".hdf5")

    est_dir = base_params['base_config']['outpath']
    if not os.path.exists(est_dir):
        os.makedirs(est_dir)
    result_base = est_par['estimation_results_base']
    estimators = est_par['estimators']
    print("list of estimators:")
    print(list(estimators.keys()))
    table = "Code        | PIT outrate |     KS      |     CvM     |  CDE loss   |    sigma    |    bias     | fout \n"
    for est_key in estimators:
        if est_par['run_estimation']:
            est_train_data = input_train.copy()
            est_test_data = input_test.copy()
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
                pz.inform(est_train_data)  # train on training data

            # don't bother with iterator for now, just run the entire chunk
            pz_data = pz.estimate(est_test_data)

            # save data?
            if est_par['save_pdfs']:
                outfile = result_base + f"{name}.pq"
                pz_data.write_to(os.path.join(est_dir, outfile))

            print(f"finished estimation using code {name}")
        else:
            # if run_estimation is False, then try loading saved results file
            # from a previous run of goldenspike where run_estimation was True
            name = estimators[est_key]['run_params']['class_name']
            infile = result_base + f"{name}.pq"
            res_file = os.path.join(est_dir, infile)
            pz_data = qp.read(res_file)
            
        # ####Evaluation
        if eval_par['run_evaluation']:
            eval_dir = eval_par['evaluation_directory']
            if not os.path.exists(eval_dir):
                print(f"directory {eval_dir} doesn't exist, creating...")
                os.makedirs(eval_dir)

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
                        code=f"{name}", pit_out_rate=pit_out_rate, outdir=figdir,
                        savefig=True)

            if eval_par['save_pit_mode_vals']:
                pit_file = os.path.join(eval_dir, f"{name}_pit_mode_values.pq")
                pitdict = {'pit': pit_vals, 'photoz_mode': z_mode}
                pit_df = pd.DataFrame(pitdict)
                tables_io.io.write(pit_df, pit_file)

        if eval_par['run_evaluation']:
            res_file = os.path.join(eval_dir, eval_par['results_file'])
            with open(res_file, "w") as f:
                f.write(table)


if __name__ == "__main__":
    main()
