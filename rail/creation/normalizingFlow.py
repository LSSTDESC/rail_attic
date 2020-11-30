"""
Wrapper for a Normalizing Flow built with Flax:
https://flax.readthedocs.io/en/latest/notebooks/flax_guided_tour.html

For an intro to normalizing flows, see this two part tutorial:
https://blog.evjang.com/2018/01/nf1.html

Good intro sources in the literature:
[1]: Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear
     Independent Components Estimation. _arXiv preprint arXiv:1410.8516_,
     2014. https://arxiv.org/abs/1410.8516
[2]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
     using Real NVP. In _International Conference on Learning
     Representations_, 2017. https://arxiv.org/abs/1605.08803
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax import nn
import dill
import pandas as pd
from generator import Generator


class NormalizingFlow(Generator):
    """
    Wrapper class for flax normalizing flows.
    
    Input:
    module - a flax module that returns a tfd.TransformedDistribution object
    hyperparams - a dictionary containing the hyperparameters for whatever module
                    you're using. The only mandatory hyperparameter is 'nfeatures'.
    params - the model parameters. If not provided, random parameters are generated
    transform_data - the transformation that maps from the data space to the flow input.
                    If none provided, transform_data is just the identity.
    inv_transform_data - the transformation that maps from flow output to the data space.
                    If none provided, inv_transform_data is just the identity.
    file - a file to load the other arguments from (except module, which must be provided).
                    If file is provided, those other arguments are ignored.
                    
    Methods:
    log_prob(x) - calculates the log probability that x is drawn from the
                    transformed probability distribution
    sample(n_samples, seed=0) - draws n_samples random samples from the distribution
                    seed sets the random seed.
    train(trainingset, testset=None, niter=2000, batch_size=1024, seed=None, 
                    return_losses=False, verbose) - 
                    trains the normalizing flow on the given training set (which must be 
                    in the form of a pandas dataframe) and updates the model parameters.
                    testset is an optional separate data set to evaluate a validation loss.
                    niter is the number of training iterations.
                    batch_size is the size of batches to train on. 
                    seed is a random seed for drawing batches.
                    If return_losses is True, returns a list of the training losses.
                    If verbose is True, training loss will be printed everytime 5% of the
                        iterations have been completed.
    save(file) - saves the hyperparams, params, and data tranformations to a file such that
                    the flow can be reloaded by instantiating with the file argument.
    """
    
    def __init__(self, module, hyperparams=None, params=None, transform_data=None, inv_transform_data=None, file=None):

        # make sure we either get the list of hyperparameters or a file to load from
        if hyperparams is None and file is None:
            raise ValueError('User must pass either hyperparams or file during instantiation')

        # warn that other arguments are ignored if a file is passed
        if file and any((hyperparams, params, transform_data, inv_transform_data)):
            print("Warning: ignoring passed arguments in favor of values stored in file")

        # if a file is passed, load all the arguments from there
        if file:
            with open(file, 'rb') as handle:
                save_dict = dill.load(handle)
            hyperparams = save_dict['hyperparams']
            params = save_dict['params']
            transform_data = save_dict['transform_data']
            inv_transform_data = save_dict['inv_transform_data']
        
        # make sure the hyperparameter dict contains nfeatures
        if 'nfeatures' not in hyperparams:
            raise KeyError('nfeatures must be in the hyperparameter dictionary')
        
        # save the hyperparameters
        self.hyperparams = hyperparams.copy()
        
        # create flax modules
        @nn.module
        def forward_module(x):
            return module(hyperparams).bijector.forward(x)
        @nn.module
        def inverse_module(x):
            return module(hyperparams).bijector.inverse(x)
        @nn.module
        def sampler_module(n_samples, seed):
            return module(hyperparams).sample(n_samples, seed)
        @nn.module
        def log_prob_module(x):
            return module(hyperparams).log_prob(x)

        # Normalizing flow parameters:
        # if no params provided, create some random ones
        if params is None:
            dummy_input = jnp.zeros((1,hyperparams['nfeatures']))
            _, params = log_prob_module.init(jax.random.PRNGKey(0), dummy_input)
        self.params = params

        # instantiate the flax models, based on the above modules, with these params
        self._forward = nn.Model(forward_module, params)
        self._inverse = nn.Model(inverse_module, params)
        self._sampler = nn.Model(sampler_module, params)
        self._log_prob = nn.Model(log_prob_module, params)

        # save the functions for data transformation
        self.transform_data = transform_data
        self.inv_transform_data = inv_transform_data
        
        
    def forward(self, y):
        x = self._forward(y)
        if self.inv_transform_data:
            x = self.inv_transform_data(x)
        return x

    def inverse(self, x):
        trans_x = x
        if self.transform_data:
            trans_x = self.transform_data(trans_x)
        return self._inverse(trans_x.iloc[:,:self.hyperparams['nfeatures']])
        
    def sample(self, n_samples, seed=None):
        seed = np.random.randint(1e18) if seed is None else seed
        samples = self._sampler(n_samples, jax.random.PRNGKey(seed))
        if self.inv_transform_data:
            samples = self.inv_transform_data(samples)
        return samples
    
    def log_prob(self, x, convolve_err=False):
        trans_x = x
        if self.transform_data:
            trans_x = self.transform_data(trans_x, transform_errs=convolve_err)
        if convolve_err:
            return self._log_prob_convolved(trans_x, self._inverse)
        else:
            return self._log_prob(trans_x.iloc[:,:self.hyperparams['nfeatures']])

    def _log_prob_convolved(self, x, f_inverse):

        nfeatures = self.hyperparams['nfeatures']
        data, errs = jnp.array(x)[:,:nfeatures], jnp.array(x)[:,nfeatures:]
        
        def log_prob_convolved(data, errs):
            Jinv = jax.jacfwd(f_inverse)(data[None,:]).squeeze()
            IsigmaY = jnp.identity(nfeatures) + Jinv @ jnp.diag(errs) @ Jinv.T
            Y = f_inverse(data[None,:]).squeeze()
            return jnp.linalg.slogdet(Jinv)[1] - 1/2 * jnp.linalg.slogdet(2*jnp.pi*IsigmaY)[1] - 1/2 * Y.T @ jnp.linalg.solve(IsigmaY,Y)

        return jax.vmap(log_prob_convolved)(data, errs)
    
    def pz_estimate(self, data, convolve_err=False, zmin=0, zmax=4, dz=0.01):
        
        # generate the redshift grid
        zs = np.arange(zmin, zmax+dz, dz)
        
        cols = self.hyperparams['data_cols']
        if convolve_err:
            for col in cols:
                if col != 'redshift':
                    cols = np.append(cols, f'{col}err')

        # log_prob for each galaxy in the sample
        log_prob = self.log_prob(pd.DataFrame({col: np.repeat(data[col], len(zs)) if col not in ('redshift','redshift_err')
                                                else np.tile(zs, len(data)) if col == 'redshift'
                                                else np.zeros(len(data)*len(zs)) for col in cols}),
                                convolve_err=convolve_err)

        # reshape so each row is a galaxy
        log_prob = log_prob.reshape((len(data), -1))
        # exponentiate 
        pdfs = np.exp(log_prob)
        # normalize
        pdfs = pdfs / (pdfs * dz).sum(axis=1).reshape(-1,1)
        
        return pdfs
    
    def train(self, trainingset, testset=None, niter=2000, batch_size=1024, seed=None, 
              return_losses=False, verbose=False):
        
        # save the titles of the data columns
        self.hyperparams['data_cols'] = trainingset.columns.values

        # the optimizer used for training
        if 'learning_rate' in self.hyperparams:
            learning_rate = self.hyperparams['learning_rate']
        else:
            learning_rate = 0.001
        optimizer = flax.optim.Adam(learning_rate=learning_rate).create(self._log_prob)
        
        # compile a function that does a single training step
        @jax.jit
        def train_step(optimizer, batch):

            def loss_fn(model):
                log_prob = model(batch)
                return -jnp.mean(log_prob)

            loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
            optimizer = optimizer.apply_gradient(grad)
            return loss, optimizer
        
        # loop through the training
        losses = []
        testlosses = []
        np.random.seed(seed)
        for i in range(niter):
            
            # get a batch of the trainingset and transform it
            batch = trainingset.sample(n=batch_size, replace=False)
            if self.transform_data:
                batch = self.transform_data(batch, transform_errs=False)
                
            # do a step of the training
            loss, optimizer = train_step(optimizer, jnp.array(batch))
            losses.append(loss)
            
            # every 5% of iterations
            if i % int(0.05*niter) == 0 or i == niter-1:
                # print the training loss
                if verbose:
                    print(loss)
                # if a testset is provided, evaluate training loss
                if testset is not None and (verbose or return_losses):
                    testbatch = testset.sample(n=batch_size, replace=False)
                    if self.transform_data:
                        testbatch = self.transform_data(testbatch, transform_errs=False)
                    testlosses.append(-np.mean(optimizer.target(testbatch)))
        
        # update the parameters
        self._log_prob = self._log_prob.replace(params=optimizer.target.params)
        self._sampler = self._sampler.replace(params=optimizer.target.params)
        self.params = optimizer.target.params
        
        # return list of training losses if true
        if return_losses and testset is None:
            return losses
        elif return_losses:
            return losses, testlosses

    def save(self, file):

        save_dict = {'hyperparams' : self.hyperparams,
            'params' : self.params,
            'transform_data' : self.transform_data,
            'inv_transform_data' : self.inv_transform_data}

        with open(file, 'wb') as handle:
            dill.dump(save_dict, handle, recurse=True)