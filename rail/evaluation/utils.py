from collections import NamedTuple

# These generic mathematical metrics will be moved to qp at some point.
stat_and_pval = NamedTuple('stat_and_pval', ['statistic', 'p_value'])
stat_crit_sig = NamedTuple('stat_crit_sig', ['statistic', 'critical_values', 'significance_level'])
