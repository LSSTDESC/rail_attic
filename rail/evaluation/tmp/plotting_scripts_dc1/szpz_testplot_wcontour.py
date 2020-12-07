#quick test of multipanel plots
#Oct 11, 2017

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import scipy.stats as sps
import seaborn as sbn


def main(argv):
    numcodes = 10
    numgals = 5000
    codenames = ["BPZ","EAZY","LePhare","GPz","ANNz2","NN","CDESB","METAPhoR","Skynet","TPZ"]
    sigmas = np.array([0.02,0.015,0.020,0.011,0.015,0.02,0.013,0.04,0.05,0.06])

    xsz = sps.uniform.rvs(size=numgals)
    sz = 2.0*xsz
    pzs = np.zeros([numcodes,numgals])
    #randos = np.reshape(sps.norm.rvs(loc=0.0,scale=0.2,size=numcodes*numgals),
                        #(numcodes,numgals))

    for i in range(numcodes):
        randos = sps.norm.rvs(loc=0.0,scale=np.sqrt(sigmas[i]),size=numgals),
        pzs[i,:] = sz + randos

    numrows = 4
    numcols = 3
    
    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12,16))
    fig.subplots_adjust(hspace=0.0)
    fig.subplots_adjust(wspace=0.0)
    #fig.tight_layout()


    for i in range(numcodes):
        ax = plt.subplot(numrows,numcols,i+1)
        ax.scatter(sz,pzs[i],marker='.',c='r',edgecolor='none',lw=1,label=codenames[i])
#        sbn.kdeplot(sz, pzs[i], shade=True, shade_lowest=False, 
#                    levels=[.5, .6, .7, .8, .9, 1.0, 1.1, 1.2], cmap="Reds")
        sbn.kdeplot(sz, pzs[i], shade=True, shade_lowest=False, 
                    n_levels=7, cmap="Reds")
        ax.set_xlim([0.0,1.99])
        ax.set_ylim([0.0,1.99])
        ax.legend(loc="upper left",scatterpoints=1)
#add lines that show sigma boundary
        finsigma = np.maximum(3.0*sigmas[i],0.06)
        upper_l = finsigma
        upper_h = 2.0 + 3.0*(finsigma)
        lower_l = -1.*finsigma
        lower_h = 2.0 - 3.0*(finsigma)
        ax.plot([0.,2.],[upper_l,upper_h],c='c',lw=2,linestyle='--')
        ax.plot([0.,2.],[lower_l,lower_h],c='c',lw=2,linestyle='--')
        ax.plot([0.,2.],[0.,2.],c='k',lw=2)
        if i%3==0:
            ax.set_ylabel("$photo-z$",fontsize=15)
        else:
            ax.set_yticklabels([])
        if i<7:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("$spec-z$",fontsize=15)
    #xx = fig.add_subplot(numcols,numrows,11)
    #xx.axis('off')
    #xx = fig.add_subplot(numcols,numrows,12)
    #xx.axis('off')
    plt.savefig("testplot.png",format='png')
    plt.show()

if __name__=="__main__":
    main(sys.argv)
