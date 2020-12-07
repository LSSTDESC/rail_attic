#script to calculate PIT values, QQ vectors, KS, CvM, and AD tests for 
#BPZ, as a test of what we'll do at NERSC

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
    print "calculating PIT vals..."
    bpzPIT = bpzobj.PIT()
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)

    #print "PIT!"
    #print bpzPIT
    #write to file
    outfp = open("TESTPITVALS.out","w")
    outfp.write("#ID PIT\n")
    for i in range(len(ID)):
        outfp.write("%d %0.5f\n"%(ID[i],bpzobj.pitarray[i]))
    outfp.close()
#QQplot
    print "making QQ plot..."
    qq_qtheory,qq_qdata = bpzobj.QQvectors(using='gridded',dx=0.0001,Nquants=1001)
    outfp = open("TESTQQvectors.out","w")
    outfp.write("#qtheory qdata\n")
    for i in range(len(qq_qtheory)):
        outfp.write("%0.6f %0.6f\n"%(qq_qtheory[i],qq_qdata[i]))
    outfp.close()
    
###all stats
    outfp = open("TEST_STATS_KSCVMAD.out","w")

    ksstat,kspval = bpzobj.KS(using='gridded',dx=0.0001)
    outfp.write("KSval: %.6g\n"%(ksstat))
    outfp.write("KSpval: %.6g\n"%(kspval))

    cvmstat,cvmpval = bpzobj.CvM(using='gridded',dx=0.0001)
    outfp.write("CvMval: %.6g\n"%(cvmstat))
    outfp.write("Cvmpval: %.6g\n"%(cvmpval))

    vmn = 0.05
    vmx = 0.95
    adstat,adpval = bpzobj.AD(using='gridded',dx=0.0001,vmin=vmn,vmax=vmx)
    outfp.write("ADval for vmin/vmax=%.3f %.3f: %.6g\n"%(vmn,vmx,adstat))
    outfp.write("ADpval: %.6g\n"%(adpval))


    vmn = 0.1
    vmx = 0.9
    adstat,adpval = bpzobj.AD(using='gridded',dx=0.0001,vmin=vmn,vmax=vmx)
    outfp.write("ADval for vmin/vmax=%.3f %.3f: %.6g\n"%(vmn,vmx,adstat))
    outfp.write("ADpval: %.6g\n"%(adpval))


    vmn = 0.01
    vmx = 0.99
    adstat,adpval = bpzobj.AD(using='gridded',dx=0.0001,vmin=vmn,vmax=vmx)
    outfp.write("ADval for vmin/vmax=%.3f %.3f: %.6g\n"%(vmn,vmx,adstat))
    outfp.write("ADpval: %.6g\n"%(adpval))


    print "finished\n"
if __name__ == "__main__":
    main(sys.argv)
