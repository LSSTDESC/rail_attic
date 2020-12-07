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
    
    qqvectorfile = "TESTQQvectors.out"
    
    numcodes = len(codes)
    numquants = 1001 #number of zbins in each NZPLOT_vectors.out file
    qtvals = np.zeros([numcodes,numquants])
    qdvals = np.zeros([numcodes,numquants])

    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,qqvectorfile)
        #print fullpath
        data = np.loadtxt(fullpath,skiprows=1)
        qtvals[i,:] = data[:,0]
        qdvals[i,:] = data[:,1]
        print "read in data for %s"%xfile
        
  
    numrows = 2
    numcols = 4

    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12,6))
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.0)
    
    for i in range(numcodes):
        ax = plt.subplot(numrows,numcols,i+1)
        tmplabel = "%s"%(labels[i])
        plt.plot(qtvals[i],qdvals[i],c='r',linestyle='-',linewidth=3,label=tmplabel)
        plt.plot([0,1],[0,1],color='k',linestyle='-',linewidth=1)

        ax.set_xlim([0.0,0.99])
        ax.set_ylim([0.0,0.99])
        ax.legend(loc="upper left",fontsize=12)
        if (i<(numcodes-5)):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Qtheory",fontsize=18)
        if i%4==0:
            ax.set_ylabel("Qdata",fontsize=18)
        else:
            ax.set_yticklabels([])

   
    plt.savefig("QQplot_10codes.jpg",format='jpg')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
