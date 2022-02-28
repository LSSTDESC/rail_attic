"""Tuples that capture standard statisical quantities """

from collections import namedtuple

# These generic mathematical metrics will be moved to qp at some point.
stat_and_pval = namedtuple('stat_and_pval', ['statistic', 'p_value'])
stat_crit_sig = namedtuple('stat_crit_sig', ['statistic', 'critical_values', 'significance_level'])
