#file to read in all the point pz data
#Oct 12, 2017
import sys,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

def main(argv):
    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY/NEWSCATMAG","FLEXZ/MAR5RESULTS","GPZ/AUG2018","LEPHARE","METAPHOR","NN/DEC8RESULTS","SKYNET","TPZ","NULL")
    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare","METAPhoR","CMNN","SkyNet","TPZ","TrainZ")
    labeldict = dict(zip(codes,labels))
    
    qqvectorfile = "TESTQQvectors.out"

    PITfile = "TESTPITVALS.out"
    
    gridindeces = np.array([0,1,2,3,8,9,10,11,16,17,18,19])
    
    numcodes = len(codes)
    numquants = 1001 #number of zbins in each NZPLOT_vectors.out file
    numgals = 399356 #number of galaxies in each file
    xlevel = float(numgals/101.) #flat line definition for PIT hist
    trainz_2sig = 446.5# 2sigma 223.27#1sigma
    xlevel_low = xlevel-trainz_2sig
    xlevel_high = xlevel+trainz_2sig
    
    pitvals = np.zeros([numcodes,numgals])
    qtvals = np.zeros([numcodes,numquants])
    qdvals = np.zeros([numcodes,numquants])
    diffvals = np.zeros([numcodes,numquants])

    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,qqvectorfile)
        #print fullpath
        data = np.loadtxt(fullpath,skiprows=1)
        qtvals[i,:] = data[:,0]
        qdvals[i,:] = data[:,1]
        diffvals[i,:] = np.subtract(data[:,1],data[:,0])
        print ("read in QQ data for %s"%xfile)
        pitpath = os.path.join(direcpath,PITfile)
        pitdata = np.loadtxt(pitpath,skiprows=1)
        pitvals[i,:] = pitdata[:,1]
        print ("read in PIT data for %s"%xfile)
  
    numrows = 3
    numcols = 4
    fig = plt.figure(figsize=(12,14))
    #fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12,9))
    #fig.subplots_adjust(hspace=0.03, wspace=0.055)
    ##sns.set()
    #gs = gridspec.GridSpec(6,4,height_ratios=[2,1,2,1,2,1],hspace=0.07,wspace=0.1)
    gs = gridspec.GridSpec(6,4,height_ratios=[1.8,1,1.8,1,1.8,1],hspace=0.1,wspace=0.125)

    #    for i in range(numcodes):
    for xx,i in enumerate(gridindeces):
        j=i
        k=i+4
        ax = plt.subplot(gs[j])
        tmplabel = "%s"%(labels[xx])
        #plt.plot(qtvals[xx],qdvals[xx],c=(0.,.287,0.287),linestyle='-',linewidth=3,
        #         label=tmplabel)
        plt.plot(qtvals[xx],qdvals[xx],c='r',linestyle='-',linewidth=3,
                 label=tmplabel)
        plt.plot([0,1],[0,1],color='k',linestyle='--',linewidth=1)

        #ax.set_xlim([0.0,0.99])
        ax.set_xlim([-0.01,1.01])
        ax.set_ylim([0.0,0.99])
        ax.legend(loc="upper left",fontsize=14,handletextpad=0.0,
                  handlelength=0)
        ax.set_xticklabels([])
        ax.tick_params(axis='both',which='major', labelsize=14,direction='in')
        ax2 = ax.twinx()
        ax2.tick_params(axis='both',direction='in',labelsize=14)
        #ax2.set_xticks([0.0,0.25,0.75,1.0])
        ax2.set_ylim([0.0,13000.])
        #bx2.set_yticklabels(['',0,2500,5000,7500,10000])
        sns.distplot(pitvals[xx], kde=False, label=tmplabel,
                     bins=np.arange(0.0, 1.01, 0.01), color='b',
                     hist_kws={"histtype": "stepfilled","alpha":0.4})
#        sns.distplot(pitvals[xx], kde=False, label=tmplabel,
#                     bins=np.arange(0.0, 1.01, 0.01), color=(.57,0,0),
#                     hist_kws={"histtype": "stepfilled"})
        plt.plot([0,1],[xlevel,xlevel],color='k',linestyle='-',linewidth=1)
        tmpx = np.linspace(0.,1.,101)
        tmpylow = np.full(len(tmpx),xlevel_low)
        tmpyhigh = np.full(len(tmpx),xlevel_high)
        plt.fill_between(tmpx,tmpylow,tmpyhigh,color="darkgray",alpha=0.75)
        #ax.set_yticklabels(['',0.0,0.2,0.4,0.6,0.8])
        bx = plt.subplot(gs[k])

        #plt.plot(qtvals[xx],diffvals[xx],c=(0.,.43,.75),linestyle='-',linewidth=3,
        #         label=tmplabel)
        plt.plot(qtvals[xx],diffvals[xx],c='r',linestyle='-',linewidth=3,
                 label=tmplabel)
        plt.plot([0,1],[0,0],color='k',linestyle='--',linewidth=1)
        bx.tick_params(axis='y',which='major', labelsize=14,direction='in')
        bx.tick_params(axis='x',direction='in')

        #bx.set_xlim([0.0,0.99])
        bx.set_xlim([-0.01,1.01])
        bx.set_ylim([-0.13,0.13])
        
        #bx2.set_xticklabels([])#set all x ticks to invisible for 2nd axes
        if i % 4 == 0:
            ax.set_ylabel(r"$Q_{data}$",fontsize=18)
            bx.set_ylabel(r"$\Delta$Q",fontsize=18)
            plt.yticks(fontsize=14)
        else:
            ax.set_yticklabels([])
            bx.set_yticklabels([])
            plt.yticks(fontsize=14)
        if i > 11:
            bx.set_xlabel(r"$Q_{theory}$/PIT value",fontsize=18)
            plt.xticks([0.0,0.25,0.5,0.75,1.0],["0.0","0.25","0.5","0.75","1.0"],fontsize=14)
            plt.yticks(fontsize=14)

            #bx.set_xticks([0.0,0.25,0.5,0.75,1.0],["0.0","0.25","0.5","0.75","1.0"])
        else:
            bx.set_xticklabels([])
            ax2.set_xticklabels([])

        #if (i % 4 == 3 and i<14) or i==18:# or i==18:
        if (i % 4 == 3):# or i==18:

            ax2.set_ylabel(r"Number",fontsize=18)
            plt.yticks(fontsize=14)
        else:
            ax2.set_yticklabels([])
            
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

    plt.savefig("PITANDQQplot_12codes_withsigmaband_biglabels.jpg",format='jpg')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
