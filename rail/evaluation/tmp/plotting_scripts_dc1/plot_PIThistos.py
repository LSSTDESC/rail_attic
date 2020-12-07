#file to read in all the point pz data
#Oct 12, 2017
import sys,os
import numpy as np
import matplotlib.pyplot as plt


def main(argv):
    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","EAZY","FLEXZ","GPZ","METAPHOR","SKYNET","TPZ")
    labels = ("ANNz2","BPZ","EAZY","FlexZBoost","GPz","METAPhoR","SkyNet","TPZ")
    labeldict = dict(zip(codes,labels))
    
    qqvectorfile = "TESTPITVALS.out"
    
    numcodes = len(codes)
    numgals = 399356 #number of zbins in each NZPLOT_vectors.out file
    pitvals = np.zeros([numcodes,numgals])
    
    xlevel = float(numgals/101.)

    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,qqvectorfile)
        #print fullpath
        data = np.loadtxt(fullpath,skiprows=1)
        
        pitvals[i,:] = data[:,1]
        print "read in data for %s"%xfile
        

    testinf = np.all(np.isfinite(pitvals))
    print "are all values non inf?"
    print testinf

  
    numrows = 2
    numcols = 4


    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12.1,6))
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.01)
    
    for i in range(numcodes):
        ax = plt.subplot(numrows,numcols,i+1)
        tmplabel = "%s"%(labels[i])
        plt.hist(pitvals[i],normed=False,histtype='stepfilled',color='r',alpha=0.7,bins=np.arange(0.0,1.01,0.01),label=tmplabel)
        plt.plot([0,1],[xlevel,xlevel],color='k',linestyle='--',linewidth=2)


        ax.set_xlim([-0.025,1.025])
        ax.set_ylim([0.0,12000.])
        ax.legend(loc="upper center",fontsize=12)
#,handletextpad=0,handlelength=0,fancybox=True,markerscale = 0)
        if (i<(numcodes-5)):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("PIT value",fontsize=18)
        if i%4==0:
            ax.set_ylabel("Number",fontsize=18)
        else:
            ax.set_yticklabels([])

   
    plt.savefig("PIT_histogram_plot_10codes.jpg",format='jpg')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
