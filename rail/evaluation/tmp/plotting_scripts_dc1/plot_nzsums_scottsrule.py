import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#import seaborn as sns
import scipy.ndimage.filters as scifilt

def main(argv):
    """
    Sept 1, 2018: Add Scott's rule smoothing to all curves
    Scott's rule bandwidth for scipy.stats.gaussian_kde that qp uses is 
    given by   
    n**(-1./(d+4)), so n^(-.2)                         
    for 399,356 galaxies in interval [0,2] we have a bandwidth of 
    sigma = 0.07581.  Given the array spacing of the stored NZPLOT_vectors is
    0.001, this corresponds to a bandwidth of 75.8 pixels (and an extra fudge
    factor of a 0.4 that is obviously necessary to match TrainZ that I don't 
    understand)
    """

    smooth_sigma = 75.81*0.4 #see note in docstring on bandwidth choice
    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY","FLEXZ","GPZ/AUG2018","LEPHARE",
             "METAPHOR","NN","SKYNET","TPZ","NULL2")
    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare",
              "METAPhoR","CMNN","SkyNet","TPZ","TrainZ")
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
        smooth_pzvals = scifilt.gaussian_filter1d(pzvals,smooth_sigma,
                                                  mode='constant')
        print ("read in data for %s"%xfile)
        

    numrows = 3
    numcols = 4


    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12,9))
    fig.subplots_adjust(hspace=0.025, wspace=0.055)
#    sns.set()
    
    for i in range(numcodes):
        ax = plt.subplot(numrows,numcols,i+1)
        if i == 0: #only include label for first entry
            plt.plot(z_array,szvals[i],c='b',linestyle='-',linewidth=2,
                     label="zspec", alpha=0.85)
        else:
            plt.plot(z_array,szvals[i],c='b',linestyle='-',linewidth=2,
                     alpha=0.85)
        tmplabel = "%s"%(labels[i])
        plt.plot(z_array,smooth_pzvals[i],color='r',linestyle='-',
                 linewidth=2,label=tmplabel, alpha=0.85)

        ax.set_xlim([0.0,1.99])
        ax.set_ylim([0.0,1.79])
        ax.yaxis.set_ticks(np.arange(0.,1.55,0.5))
        if i ==0: #include lines only for first entry
            ax.legend(loc="upper left",fontsize=11)
        else: 
            ax.legend(loc="upper left",fontsize=11, handletextpad=0,
                      handlelength=0)
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

   
    plt.savefig("NZsumplot_12codes_scottsrule.jpg",format='jpg')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
