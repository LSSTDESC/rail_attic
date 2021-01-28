import sys
from rail.evaluation.sample import Sample
from rail.evaluation.metrics import *
#Metrics  # import metrics subclasses explicitly?

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
        python eval.py <code name> <PDFs file> <sample name> <z-spec file>

    Example:
        python eval.py FZBoost ./results/FZBoost/test_FZBoost.hdf5 toy_data ../tests/data/test_dc2_validation_9816.hdf5

    """
    print()
    print()
    print("        *** RAIL EVALUATION MODULE ***")
    print()
    if len(argv) != 5:
        print()
        print("ERROR: invalid entry")
        print()
        print("Usage:")
        print("    python eval.py <code name> <PDFs file> <sample name> <z-spec file>")
        print()
        print("Example:")
        print("    python eval.py FZBoost ./results/FZBoost/test_FZBoost.hdf5 toy_data ../tests/data/test_dc2_validation_9816.hdf5")
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
    metrics = Metrics(sample)
    metrics_table = metrics.print_summary()
    COLOCAR CDE LOSS COMO SUBCLASSE
    print()



    print("Making plots...")
    print()
    print()

    #raise NotImplementedError("Module under construction.")

if __name__ == "__main__":
    main(sys.argv)

