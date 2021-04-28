import sys
from rail.evaluation.utils import *
import time as t


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
        python evaluator.py <code name> <PDFs file> <sample name> <z-spec file>

    Example:
        python evaluator.py FlexZBoost ./results/FZBoost/test_FZBoost.hdf5 toy_data ../tests/data/test_dc2_validation_9816.hdf5

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
        print("    python evaluator.py <code name> <PDFs file> <sample name> <z-spec file>")
        print()
        print("Example:")
        print("    python evaluator.py FlexZBoost ./results/FZBoost/test_FZBoost.hdf5 toy_data ../tests/data/test_dc2_validation_9816.hdf5")
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
    print()
    print()

    print("Computing metrics...")
    pit = PIT(pdfs, zgrid, ztrue)
    pit.evaluate()
    pits = pit.metric
    summary = Summary(pdfs, zgrid, ztrue)
    summary.print_metrics_table(pits=pits)
    print()
    print()


    print("Making plots...")
    print()
    print()
    fig_filename = pit.plot_pit_qq(code=code, savefig=True)
    # TO DO: ADD METRICS PLOT HERE
    print(f"PIT-QQ plot saved as:   {fig_filename}")
    print()
    print ()

    t1 = t.time()
    dt = t1 - t0
    print(f"Done! (total time: {int(dt)} seconds)")
    print()
    print()



if __name__ == "__main__":
    main(sys.argv)

