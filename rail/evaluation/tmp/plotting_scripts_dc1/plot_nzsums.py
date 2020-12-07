#Make plot of all N(z) sums including Null aka TrainZ
#July 2018
import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main(argv):
    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY","FLEXZ","GPZ","LEPHARE","METAPHOR","NN","SKYNET","TPZ","NULL2")
    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare","METAPhoR","CMNN","SkyNet","TPZ","TrainZ")
    labeldict = dict(zip(codes,labels))
    
    nzvectorfile = "NZPLOT_vectors.out"
    
    numcodes = len(codes)
    numzs = 2001 #number of zbins in each NZPLOT_vectors.out file
    szvals = np.zeros([numcodes,numzs])
    pzvals = np.zeros([numcodes,numzs])
    z_array = np.zeros(numzs)

    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,nzvectorfile)
        #print fullpath
        data = np.loadtxt(fullpath,skiprows=1)
        if i == 0:
            z_array = data[:,0]
        szvals[i,:] = data[:,1]
        pzvals[i,:] = data[:,2]
        print "read in data for %s"%xfile
        


  
    numrows = 3
    numcols = 4


    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12,9))
    fig.subplots_adjust(hspace=0.025, wspace=0.055)
    sns.set()
    
    for i in range(numcodes):
        ax = plt.subplot(numrows,numcols,i+1)
        plt.plot(z_array,szvals[i],c='b',linestyle='-',linewidth=2,label="zspec KDE sum", alpha=0.85)
        tmplabel = "%s N(z) sum"%(labels[i])
        plt.plot(z_array,pzvals[i],color='r',linestyle='-',linewidth=2,label=tmplabel, alpha=0.85)

        ax.set_xlim([0.0,1.99])
        ax.set_ylim([0.0,1.99])
        ax.legend(loc="upper left",fontsize=11)
        if (i<(numcodes-4)):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("redshift",fontsize=18)
        if i%4==0:
            ax.set_ylabel("N(z)",fontsize=15)
        else:
            ax.set_yticklabels([])

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

   
    plt.savefig("NZsumplot_12codes.jpg",format='jpg')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
