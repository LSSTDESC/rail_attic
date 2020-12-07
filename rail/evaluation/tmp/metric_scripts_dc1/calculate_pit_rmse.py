#Read in QQ vectors, calculate RMSE 
#
import sys,os
import numpy as np
import qp
import scipy.stats as sps



def main(argv):
    """
    Quick script to calculate the RMSE for the QQ plots
    """
    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY","FLEXZ","GPZ","LEPHARE",
             "METAPHOR","NN","SKYNET","TPZ","NULL2")
    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare",
              "METAPhoR","NN","SkyNet","TPZ","TrainZ")
    labeldict = dict(zip(codes,labels))
    
    nzvectorfile = "TESTQQvectors.out"
    outfp = open("RMSE_QQ.out","w")
    
    numcodes = len(codes)
    numzs = 1001 #number of PITbins in each TESTQQvectors.out file
    trueqqvec = np.zeros([numcodes,numzs])
    stackqqvec = np.zeros([numcodes,numzs])
    z_array = np.zeros(numzs)

    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,nzvectorfile)
        data = np.loadtxt(fullpath,skiprows=1)
        trueqqvec = data[:,0]
        trueobj = qp.PDF(gridded=(trueqqvec,trueqqvec))
     
        stackqqvec = data[:,1]
        print "read in data for %s"%xfile
        
        stackobj = qp.PDF(gridded=(trueqqvec,stackqqvec))      
        xrmse = qp.utils.calculate_rmse(trueobj,stackobj,limits=(0.0,1.0),
                                        dx=0.001)
        outfp.write("%sQQRMSE: %5.5f\n"%(xfile,xrmse))
    outfp.close()
    print "finished"

if __name__=="__main__":
    main(sys.argv)
