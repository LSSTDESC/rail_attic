#script to calculate PIT values, QQ vectors, KS, CvM, and AD tests for 
#BPZ, as a test of what we'll do at NERSC

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import cde_individual_metrics as inmet
import skgof
import sys,os
import qp
import time
import ingestFlexZdata as ingdata

def main(argv):
    starttime = time.time()
    currenttime = time.time()
    outfile = "CDE_STATS.out"
    outfp = open(outfile,"w")
    z_array,ID,szs,mags,pzs = ingdata.ingestflexzdata()



    print "making Ensemble..."
    approx_pdf = qp.Ensemble(pzs.shape[0],gridded=(z_array,pzs),procs=3)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)
    print "making EvaluateMetric Object"
    bpzobj = inmet.EvaluateMetric(approx_pdf,szs)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)
    #print "calculating PIT vals..."
    #bpzPIT = bpzobj.PIT()
    #oldtime = currenttime
    #currenttime = time.time()
    #print "took %g seconds"%(currenttime-oldtime)

    #print "PIT!"
    #print bpzPIT
    #write to file
    
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)
    print "calculating cdeloss..."
    tmpxgrid = np.linspace(0.0,10.0,1000)
    cde_loss = bpzobj.cde_loss(tmpxgrid)
    print "CDE loss: %g\n"%cde_loss
    outfp.write("CDE LOSS:\n%.6g\n"%(cde_loss))
    outfp.close()

    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)
    print "finished\n"
if __name__ == "__main__":
    main(sys.argv)
