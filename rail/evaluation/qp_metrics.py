from scipy import stats


def KS(qp_ens, ref_dist):
    """ 
    Use scipy.stats.kstest to compute the Kolmogorov-Smirnov statistic for
    the PIT values by comparing with a reference distribution between 
    0 and 1.
    Parameters:
    -----------
    qp_ens:
    qp ensemble
    ref_dist: 
    scipy stats distribution
    """
        _statistic, _pvalue = stats.kstest(qp_ens.rvs, ref_dist.cdf)
        return _statistic, _pvalue


def CvM(qp_ens, ref_dist):
    """ 
    Use scipy.stats.kstest to compute the Cramer-von Mises statistic for
    the PIT values by comparing with a reference distribution between
    0 and 1.
    Parameters:
    -----------
    qp ensemble
    ref_dist:
    scipy stats distribution
    """
    _statistic, _pvalue = stats.cramervonmises(qp_ens.rvs, ref_dist.cdf)
    return _statistic, _pvalue


    

