#script to calculate Nz stack values, KS, CvM, and AD tests for 
#BPZ, as a test of what we'll do at NERSC
#Nov 1, 2017

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import individual_metrics as inmet
import skgof
import sys,os
import qp
import time
import ingestFlexZdata as ingdata

def main(argv):

    starttime = time.time()
    currenttime = time.time()
#
    z_array,ID,szs,mags,pzs = ingdata.ingestflexzdata()

    print "making Ensemble..."
    approx_pdf = qp.Ensemble(pzs.shape[0],gridded=(z_array,pzs),procs=3)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)
    print "making NzSumEvaluateMetric Object, with stacking..."
    
    nzobj = inmet.NzSumEvaluateMetric(approx_pdf,szs, eval_grid=z_array,
                                      using='gridded',dx=0.0001)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)
    print "calculating Nz sum vectors..."
    newgrid = np.arange(0.0,2.0001,0.001)
    #create qp object of samples from the spec-z sample
    szsamplepdf = qp.PDF(samples=szs)
    specznz = szsamplepdf.evaluate(newgrid,using='samples',vb=True,
                                   norm=False)[1] #only grab the 2nd part of the tuples!
    photznz = nzobj.stackpz.evaluate(newgrid,using='gridded',vb=True,
                                     norm=False)[1] #only grab the 2nd part of the tuples!
    outfp = open("NZPLOT_vectors.out","w")
    outfp.write("#z_array speczNz photzNz\n")
    for i in range(len(newgrid)):
        outfp.write("%f %g %g\n"%(newgrid[i],specznz[i],photznz[i]))
    outfp.close()
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)
    print "calculating KS stat..."

    ks_stat,ks_pval = nzobj.NZKS()
    print "ks_stat: %g\nks_pval: %g\n"%(ks_stat,ks_pval)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)

    cvm_stat,cvm_pval = nzobj.NZCVM()
    print "cvm_stat: %g\cvm_pval: %g\n"%(cvm_stat,cvm_pval)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)

    zmin = min(szs)
    zmax = max(szs)
    delv = (zmax - zmin)/200.

    ad_stat,ad_pval = nzobj.NZAD(vmin=zmin,vmax=zmax,delv=delv)
    print "ad_stat: %g\ad_pval: %g\n"%(ad_stat,ad_pval)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)


    ad_statx,ad_pvalx = nzobj.NZAD(vmin=0.0,vmax=2.0,delv=0.01)
    print "ad_stat full range: %g\ad_pval: %g\n"%(ad_statx,ad_pvalx)
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)


###all stats
    outfp = open("NZ_STATS_KSCVMAD.out","w")

    outfp.write("KSval: %.6g\n"%(ks_stat))
    outfp.write("KSpval: %.6g\n"%(ks_pval))

    outfp.write("CvMval: %.6g\n"%(cvm_stat))
    outfp.write("Cvmpval: %.6g\n"%(cvm_pval))

  
    outfp.write("ADval for vmin/vmax=%.3f %.3f: %.6g\n"%(zmin,zmax,ad_stat))
    outfp.write("ADpval: %.6g\n"%(ad_pval))


    outfp.write("ADval for vmin/vmax=0.0/2.0: %.6g\n"%(ad_statx))
    outfp.write("ADpval: %.6g\n"%(ad_pvalx))


    outfp.close()

    print "finished\n"
if __name__ == "__main__":
    main(sys.argv)
