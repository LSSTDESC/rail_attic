class baseGenerator():

    def __init__(self):

        return

    def load_model(self, model_param_file):

        """
        Load the fully trained model that can create samples and
        return probabilities for points in mag-redshift space.
        """

        raise NotImplementedError("Implemented in specific model class.")

    def return_sample(self, n_samples, seed=None, **kwargs):

        """
        Return a randomly selected sample catalogs of size n_samples.
        """

        raise NotImplementedError("Implemented in specific model class.")

    def return_log_prob(self, datapoint, **kwargs):

        """
        Return the log_prob value from the distribution
        fit to the mag-redshift space
        """

        raise NotImplementedError("Implemented in specific model class.")