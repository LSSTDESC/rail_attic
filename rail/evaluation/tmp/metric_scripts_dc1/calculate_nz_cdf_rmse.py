#Read in N(z) sum vectors, calculate CDF, compute RMSE
#June 2018
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import qp
import scipy.stats as sps


def speczCDF(szvec,zgrid):
    """calculate the empirical CDF of a distribution of spec-z values (szvec) 
    on a grid (zgrid)"""
    tot = len(szvec)
    fracvec = np.zeros(len(zgrid))
    for i,val in enumerate(zgrid):
        mask = (szvec<val)
        num = len(szvec[mask])
        frac = float(num)/float(tot)
        fracvec[i] = frac
        #print ("z=%f: %d out of %d <%f for frac %.5f"%(val,num,tot,val,frac))
    return fracvec
               


def main(argv):
    """
    Quick script to calculate the RMSE for the stacked N(z) using empiricalCDF
    rather than the values output in NZPLOTvectors.out.  This is necessary
    because qp uses KDE and Scott's rule to determine smoothing in the specz
    sample in that output, and Scott's rule is not ideal for some codes.  Using
    the eCDF instead eliminates that bandwidth choice
    """
    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY","FLEXZ",
             "GPZ","LEPHARE","METAPHOR","NN","SKYNET","TPZ","NULL2")
    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare",
              "METAPhoR","NN","SkyNet","TPZ","TrainZ")
    labeldict = dict(zip(codes,labels))
    
    nzvectorfile = "NZPLOT_vectors.out"
    outfp = open("RMSE_NZ_eCDF.out","w")
    numcodes = len(codes)
    numzs = 2001 #number of zbins in each NZPLOT_vectors.out file
    truenzvec = np.zeros([numcodes,numzs])
    stacknzvec = np.zeros([numcodes,numzs])
    z_array = np.zeros(numzs)


    szfile = "TESTDC1BPZ/BPZgold_idszmag.txt"
    szdata = np.loadtxt(szfile)
    szvec = szdata[:,1]

    for i,xfile in enumerate(codes):
        print ("working on code %s\n"%(xfile))
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,nzvectorfile)
        #print fullpath
        data = np.loadtxt(fullpath,skiprows=1)
        z_array = data[:,0]
        truezcdf = speczCDF(szvec,z_array)
        trueobj = qp.PDF(gridded=(z_array,truezcdf))

        stacknzvec = data[:,2]
        print "read in data for %s"%xfile
        cumstack = np.cumsum(stacknzvec)

        stackobj = qp.PDF(gridded=(z_array,cumstack))
        xrmse = qp.utils.calculate_rmse(trueobj,stackobj,limits=(0.0,2.0),
                                        dx=0.001)
        outfp.write("%sN(z)RMSE: %6.6f\n"%(xfile,xrmse))
    outfp.close()
    print "finished"

if __name__=="__main__":
    main(sys.argv)
