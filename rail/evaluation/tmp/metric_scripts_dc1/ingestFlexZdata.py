#ingest the data from FlexZBoost

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sps
import individual_metrics as inmet
import skgof
import sys,os
import qp
import time

def ingestflexzdata():
    starttime = time.time()
    currenttime = time.time()
#####FILES
#    basepath = "/sandbox/sschmidt/SIMULATIONS/RISA/FINALDATACHALLENGE/SCATERR"
    basepath = "."
    infile = "1pct_Mar5Flexzgold_pz.out"
    zarrayfile = "zarrayfile.out"
    idszmagfile = "1pct_Mar5Flexzgold_idszmag.out"
###
    z_array = np.loadtxt(zarrayfile)
    print "len z_array: %d\n"%len(z_array)
    #    z_array = np.arange(0.00,2.001,0.01)
    #z_array = np.arange(0.005,1.990026,0.009975)
    #z_array = np.arange(0.0,2.001,0.01)
    print z_array
    print len(z_array)
    fullpath = os.path.join(basepath,infile)
    print "reading in p(z)'s from %s"%fullpath
    currenttime = time.time()
    pzs = np.loadtxt(fullpath)
    print len(z_array)
    print pzs.shape
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)

    
    szfullpath = os.path.join(basepath,idszmagfile)
    print "reading in i sz mag from %s\n"%szfullpath
    szdata = np.loadtxt(szfullpath)
    ID = szdata[:,0]
    szs = szdata[:,2]
    mags = szdata[:,3]
    print "num szs: %d\n"%(len(szs))
    oldtime = currenttime
    currenttime = time.time()
    print "took %g seconds"%(currenttime-oldtime)

#    mask = (magsall<25.3)
#    mags = magsall[mask]
#    szs = szsall[mask]
#    pzs = pzsall[mask]
#    ID = IDall[mask]

    print "Gold sample number: %d\n"%(len(ID))

    return z_array,ID,szs,mags,pzs
