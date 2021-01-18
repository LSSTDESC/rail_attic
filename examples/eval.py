import sys
#from rail.evaluation.sample import Sample
#from rail.evaluation.metrics import Metrics  # import metrics subclasses explicitly?

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
        python eval.py <PDFs file> <z-spec file>

    Example:
        python eval.py results/FZBoost/test_FZBoost.hdf5 ../tests/data/test_dc2_validation_9816.hdf5

    """
    print()
    print()
    print("        *** RAIL EVALUATION MODULE ***")
    print()
    if len(argv) != 3:
        print()
        print("ERROR: invalid entry")
        print()
        print("Usage:")
        print("    python eval.py <PDFs file> <z-spec file>")
        print()
        print("Example:")
        print("    python eval.py  ./results/FZBoost/test_FZBoost.hdf5 ../tests/data/test_dc2_validation_9816.hdf5")
        print()
        print()
        sys.exit()
    else:
        pdfs_file, ztrue_file = argv[1], argv[2]
        print()
        print()
        print(f"Photo-z results (PDFs file): {pdfs_file}")
        print()
        print(f"Validation/test set (z-true file): {ztrue_file}")
        print()
        print()


    print("Computing metrics...")
    print()
    print()


    print("Making plots metrics...")
    print()
    print()

    raise NotImplementedError("Module under construction.")

if __name__ == "__main__":
    main(sys.argv)

