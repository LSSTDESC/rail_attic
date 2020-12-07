#Program to compute empirical CDF of spec-z and summed p(z)
#do for the grid of spec-z values to compute the KS distance correctly!
#Nov 10, 2017
#Sam Schmidt
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sps


def main(argv):
    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","EAZY","FLEXZ","GPZ","METAPHOR","SKYNET","TPZ")
    labels = ("ANNz2","BPZ","EAZY","FlexZBoost","GPz","METAPhoR","SkyNet","TPZ")
    labeldict = dict(zip(codes,labels))
    
    nzvectorfile = "NZPLOT_vectors.out"
    szfile = "TESTDC1BPZ/BPZgold_idszmag.txt"
    tmpdata = np.loadtxt(szfile)
    szarray = np.sort(tmpdata[:,1])
    numszs = len(szarray)
    print "numszs: %d"%numszs
    yvalues = np.linspace(0.,1.,numszs)

    numcodes = len(codes)
    numzs = 2001 #number of zbins in each NZPLOT_vectors.out file
    szcum = np.zeros([numcodes,numzs])
    pzcum = np.zeros([numcodes,numzs])
    z_array = np.zeros(numzs)

    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,nzvectorfile)
        #print fullpath
        data = np.loadtxt(fullpath,skiprows=1)
        if i == 0:
            z_array = data[:,0]
        tmpszarr = data[:,1]
        tmpcum = np.cumsum(tmpszarr)
        szcum[i,:] = tmpcum/tmpcum[-1] #normalize by dividing by last element
        tmppzarr = data[:,2]
        tmppcum = np.cumsum(tmppzarr)
        pzcum[i,:] = tmppcum/tmppcum[-1]
        print "read in data for %s"%xfile
        


  
    numrows = 2
    numcols = 4


    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12,6))
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.0)
    
    for i in range(numcodes):
        #create 1d iterp for the pzcum eCDF, evaluate it on the grid of sz
        #values
        interpfunc = sps.interp1d(z_array,pzcum[i],kind='linear',bounds_error=False,fill_value=0.0)
        pzcumvec = np.zeros(numszs) 
        for j,xsz in enumerate(szarray):
            pzcumvec[j] = interpfunc(xsz)

        tmpks = np.amax(np.abs(yvalues - pzcumvec))
        print "KS val for code %s is %.5f\n"%(labels[i],tmpks)
        ax = plt.subplot(numrows,numcols,i+1)
        plt.plot(szarray,yvalues,c='b',linestyle='-',linewidth=3,label="zspec eCDF")
        tmplabel = "%s eCDF\nKS: %.4f"%(labels[i],tmpks)
        plt.plot(szarray,pzcumvec,color='r',linestyle='-',linewidth=2,label=tmplabel)
      
        ax.set_xlim([0.0,1.99])
        ax.set_ylim([0.0,1.75])
        ax.legend(loc="upper left",fontsize=12)
        if (i<(numcodes-5)):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("redshift",fontsize=18)
        if i%4==0:
            ax.set_ylabel("N(z) eCDF",fontsize=15)
        else:
            ax.set_yticklabels([])

   
    plt.savefig("NZ_eCDF_plot_10codes.jpg",format='jpg')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
