import sys

import qp

from examples.evaluation.utils import *
from rail.evaluation.metrics.pit import *
from rail.evaluation.metrics.cdeloss import *
import time as t

class Summary:
    """ Summary tables with all metrics available. """
    def __init__(self, pdfs, xvals, ztrue):
        """Class constructor."""
        # placeholders for metrics to be calculated
        self._pdfs = pdfs
        self._xvals = xvals
        self._ztrue = ztrue
        self._pit_out_rate = None
        self._ks = None
        self._cvm = None
        self._ad = None
        self._cde_loss = None
        self._fzdata = qp.Ensemble(qp.interp, data=dict(xvals=xvals, yvals=pdfs))

    def evaluate_all(self, pitobj=None):
        if pitobj is None:
            pitobj = PIT(self._fzdata, self._ztrue)
        spl_ens, metamets = pitobj.evaluate()
        pit_vals = pitobj._pit_samps
        self._pit_out_rate = PITOutRate(pit_vals, spl_ens).evaluate()
        ksobj = PITKS(pit_vals, spl_ens)
        self._ks = ksobj.evaluate().statistic
        cvmobj = PITCvM(pit_vals, spl_ens)
        self._cvm = cvmobj.evaluate().statistic
        adobj = PITAD(pit_vals, spl_ens)
        self._ad = adobj.evaluate().statistic
        cdeobj = CDELoss(self._fzdata, self._xvals, self._ztrue)
        self._cde_loss = cdeobj.evaluate().statistic

    def markdown_metrics_table(self, show_dc1=None, pitobj=None):
        self.evaluate_all(pitobj=pitobj)
        if show_dc1:
            dc1 = DC1()
            if show_dc1 not in dc1.codes:
                raise ValueError(f"{show_dc1} not in the list of codes from DC1: {dc1.codes}" )
            table = str("Metric|Value|DC1 reference value \n ---|---:|---: \n ")
            table += f"PIT out rate | {self._pit_out_rate:11.4f} |{dc1.results['PIT out rate'][show_dc1]:11.4f} \n"
            table += f"KS           | {self._ks:11.4f}  |{dc1.results['KS'][show_dc1]:11.4f} \n"
            table += f"CvM          | {self._cvm:11.4f} |{dc1.results['CvM'][show_dc1]:11.4f} \n"
            table += f"AD           | {self._ad:11.4f}  |{dc1.results['AD'][show_dc1]:11.4f} \n"
            table += f"CDE loss     | {self._cde_loss:11.2f} |{dc1.results['CDE loss'][show_dc1]:11.2f} \n"
        else:
            table = "Metric|Value \n ---|---: \n "
            table += f"PIT out rate | {self._pit_out_rate:11.4f} \n"
            table += f"KS           | {self._ks:11.4f}  \n"
            table += f"CvM          | {self._cvm:11.4f} \n"
            table += f"AD           | {self._ad:11.4f}  \n"
            table += f"CDE loss     | {self._cde_loss:11.2f} \n"
        return Markdown(table)

    def print_metrics_table(self, pitobj=None):
        self.evaluate_all(pitobj=pitobj)
        table = str(
            "   Metric    |    Value \n" +
            "-------------|-------------\n" +
            f"PIT out rate | {self._pit_out_rate:11.4f}\n" +
            f"KS           | {self._ks:11.4f}\n" +
            f"CvM          | {self._cvm:11.4f}\n" +
            f"AD           | {self._ad:11.4f}\n" +
            f"CDE loss     | {self._cde_loss:11.4f}\n" )
        print(table)

def main(argv):
    """ RAIL Evaluation module - command line mode:
    * Compute all metrics available and display them in a table.
    * Make PIT-QQ plot and save it a PNG file.

    Parameters:
    -----------
    argv: `sys.argv`, `list`
        list of parameters inserted on command line

    Usage:
    ------
        python main.py <code name> <PDFs file> <sample name> <z-spec file>

    Example:
        python main.py FlexZBoost ../estimation/results/FZBoost/test_FZBoost.hdf5 toy_data ../../tests/data/test_dc2_validation_9816.hdf5

    """
    t0 = t.time()
    print()
    print()
    print("        *** RAIL EVALUATION MODULE ***")
    print()
    if len(argv) != 5:
        print()
        print()
        print("Usage:")
        print("    python main.py <code name> <PDFs file> <sample name> <z-spec file>")
        print()
        print("Example:")
        print("    python main.py FlexZBoost ../estimation/results/FZBoost/test_FZBoost.hdf5 toy_data ../../tests/data/test_dc2_validation_9816.hdf5")
        print()
        print()
        sys.exit()
    else:
        code, pdfs_file, name, ztrue_file = argv[1], argv[2], argv[3], argv[4]
        print()
        print()
        print(f"Photo-z results by: {code}")
        print(f"PDFs file: {pdfs_file}")
        print()
        print(f"Validation/test set: {name}")
        print(f"z-true file: {ztrue_file}")
        print()
        print()

    print("Reading data...")
    pdfs, zgrid, ztrue, photoz_mode = read_pz_output(pdfs_file, ztrue_file)
    fzdata = qp.Ensemble(qp.interp, data=dict(xvals=zgrid, yvals=pdfs))
    print()
    print()

    print("Computing metrics...")
    pitobj = PIT(fzdata, ztrue)
    spl_ens, metamets = pitobj.evaluate()
    pit_vals = pitobj._pit_samps
    summary = Summary(pdfs, zgrid, ztrue)
    summary.print_metrics_table(pitobj=pitobj)
    print()
    print()


    print("Making plots...")
    print()
    print()
    pit_out_rate = PITOutRate(pit_vals, spl_ens).evaluate()
    fig_filename = plot_pit_qq(pdfs, zgrid, ztrue,
                               pit_out_rate=pit_out_rate,
                               code=code, savefig=True)
    print(f"PIT-QQ plot saved as:   {fig_filename}")
    print()
    print()

    t1 = t.time()
    dt = t1 - t0
    print(f"Done! (total time: {int(dt)} seconds)")
    print()
    print()



if __name__ == "__main__":
    main(sys.argv)
