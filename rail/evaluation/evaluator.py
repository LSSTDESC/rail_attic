import sys
from rail.evaluation.sample import Sample
from rail.evaluation.metrics import Metrics


def main(argv):
    """ RAIL Evaluation module - command line mode:
    * Compute all metrics available and display them in a table.
    * Make validation plots and save them in PNG files.

    Parameters:
    -----------
    argv: `sys.argv`, `list`
        list of parameters inserted on command line

    Usage:
    ------
        python evaluator.py <code name> <PDFs file> <sample name> <z-spec file>

    Example:
        python evaluator.py FZBoost ./results/FZBoost/test_FZBoost.hdf5 toy_data ../tests/data/test_dc2_validation_9816.hdf5

    """
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
        print("    python evaluator.py FZBoost ./results/FZBoost/test_FZBoost.hdf5 toy_data ../tests/data/test_dc2_validation_9816.hdf5")
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
    print()
    sample = Sample(pdfs_file, ztrue_file, code ="FZBoost", name="toy data")
    print(sample)
    print()
    print()

    print("Computing metrics...")
    print()
    print()
    metrics = Metrics(sample)
    metrics.print_summary()
    print()



    print("Making plots...")
    print()
    print()
    fig_filename = metrics.plot_pit_qq(savefig=True)
    # TO DO: ADD METRICS PLOT HERE
    print("Plots saved in:")
    print ()
    print (fig_filename)
    print ()



    print()
    print("Done!")
    print()
    print()


if __name__ == "__main__":
    main(sys.argv)

