#calculate moments of stacked N(z) distribution
#
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import qp
import scipy.stats as sps

def test_moments():
    f = sps.norm(loc=3,scale=1)#Normal with mu=3 sigma=1
    dx=0.02
    grid = np.arange(-2.,8.00002,dx)
    fx = f.pdf(grid) #spit out the values of the normal on a grid, make a qp object
    testobj = qp.PDF(gridded=(grid,fx))
    truth=np.array([3.0,10.0,36.0,138.0,558.0]) #values for what first five moments should be for this Normal
    for j in range(5):
        tmpmom = qp.utils.calculate_moment(testobj,j+1,using="gridded",dx=0.0001)
        print "%d %5.5f %.1f"%(j,tmpmom,truth[j])
    return
    


def main(argv):
    test_moments()
    basepathpart = "/sandbox/sschmidt/DESC/TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY/NEWSCATMAG","FLEXZ/MAR5RESULTS","GPZ","LEPHARE","METAPHOR","NN","SKYNET","TPZ","NULL")

    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare","METAPhoR","NN","SkyNet","TPZ","TrainZ")
    labeldict = dict(zip(codes,labels))
    
    nzvectorfile = "NZPLOT_vectors.out"
    outfp = open("MOMENTS_NZ.out","w")
    
    numcodes = len(codes)
    numzs = 2001 #number of zbins in each NZPLOT_vectors.out file
    truenzvec = np.zeros([numcodes,numzs])
    stacknzvec = np.zeros([numcodes,numzs])
    z_array = np.zeros(numzs)

    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,nzvectorfile)
        #print fullpath
        data = np.loadtxt(fullpath,skiprows=1)
        if i == 0:
            z_array = data[:,0]
            truenzvec = data[:,1]
            trueobj = qp.PDF(gridded=(z_array,truenzvec))
            outfp.write("####TRUE N(z) moments N=1-5\n")
            for j in range(5):
                tmpmoment = qp.utils.calculate_moment(trueobj,j+1,using="gridded",dx=0.001)
                outfp.write("%3.3f "%(tmpmoment))
            outfp.write("\n")

        stacknzvec = data[:,2]
        print "read in data for %s"%xfile
        
        stackobj = qp.PDF(gridded=(z_array,stacknzvec))
        outfp.write("####%s N(z) moments for N=1-5\n"%(xfile))
        for j in range(5):
            tmpmoment = qp.utils.calculate_moment(stackobj,j+1,using="gridded",dx=0.001)
            outfp.write("%3.3f "%(tmpmoment))
        outfp.write("\n")
    outfp.close()
    print "finished"

if __name__=="__main__":
    main(sys.argv)
