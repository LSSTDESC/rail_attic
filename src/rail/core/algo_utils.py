"""Utility functions to test alogrithms"""
import os
from rail.core.stage import RailStage
from rail.core.utils import RAILDIR
from rail.core.data import TableHandle
import scipy.special
sci_ver_str = scipy.__version__.split('.')


traindata = os.path.join(RAILDIR, 'rail/examples/testdata/training_100gal.hdf5')
validdata = os.path.join(RAILDIR, 'rail/examples/testdata/validation_10gal.hdf5')
DS = RailStage.data_store
DS.__class__.allow_overwrite = True


def one_algo(key, single_trainer, single_estimator, train_kwargs, estim_kwargs):
    """
    A basic test of running an estimator subclass
    Run inform, write temporary trained model to
    'tempmodelfile.tmp', run photo-z algorithm.
    Then, load temp modelfile and re-run, return
    both datasets.
    """
    DS.clear()
    training_data = DS.read_file('training_data', TableHandle, traindata)
    validation_data = DS.read_file('validation_data', TableHandle, validdata)

    if single_trainer is not None:
        train_pz = single_trainer.make_stage(**train_kwargs)
        train_pz.inform(training_data)

    pz = single_estimator.make_stage(name=key, **estim_kwargs)
    estim = pz.estimate(validation_data)
    pz_2 = None
    estim_2 = estim
    pz_3 = None
    estim_3 = estim

    copy_estim_kwargs = estim_kwargs.copy()
    model_file = copy_estim_kwargs.pop('model', 'None')

    if model_file != 'None':
        copy_estim_kwargs['model'] = model_file
        pz_2 = single_estimator.make_stage(name=f"{pz.name}_copy", **copy_estim_kwargs)
        estim_2 = pz_2.estimate(validation_data)

    if single_trainer is not None and 'model' in single_trainer.output_tags():
        copy3_estim_kwargs = estim_kwargs.copy()
        copy3_estim_kwargs['model'] = train_pz.get_handle('model')
        pz_3 = single_estimator.make_stage(name=f"{pz.name}_copy3", **copy3_estim_kwargs)
        estim_3 = pz_3.estimate(validation_data)

    os.remove(pz.get_output(pz.get_aliased_tag('output'), final_name=True))
    if pz_2 is not None:
        os.remove(pz_2.get_output(pz_2.get_aliased_tag('output'), final_name=True))

    if pz_3 is not None:
        os.remove(pz_3.get_output(pz_3.get_aliased_tag('output'), final_name=True))
    model_file = estim_kwargs.get('model', 'None')
    if model_file != 'None':
        try:
            os.remove(model_file)
        except FileNotFoundError:  #pragma: no cover
            pass
    return estim.data, estim_2.data, estim_3.data
