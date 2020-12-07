#File to make plots of all zpeak and zweight photo-z's calculated from
#Dritan's script.
#Feb 15, 2018
import sys,os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import point_metrics as pmet
import seaborn as sbn

###NOTE: Not all are sorted by objID, but this code stores the spec-z's for
###each code separately, so that should not be a problem

def main():
    matplotlib.rcParams['legend.handlelength']=0
    starttime = time.time()
    currenttime = time.time()

    basepathpart = "/sandbox/sschmidt/DESC/TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY/NEWSCATMAG","FLEXZ/MAR5RESULTS","GPZ/AUG2018","LEPHARE","METAPHOR","NN","SKYNET","TPZ","NULL")
    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare","METAPhoR","CMNN","SkyNet","TPZ","TrainZ")
    #codes = ("ANNZ2","BPZ","DELIGHT")
    #labels = ("ANNz2","BPZ","Delight")


    labeldict = dict(zip(codes,labels))
    
    outputfile = "ZPEAK_POINT_STATS_ALL.out"
    outfp = open(outputfile,"w")
    
    numcodes = len(codes)
    numgals = 399356 #number of zbins in each NZPLOT_vectors.out file
    sz = np.zeros([numcodes,numgals])
    zpeak = np.zeros([numcodes,numgals])
    zweight = np.zeros([numcodes,numgals])
    imag = np.zeros([numcodes,numgals])
    sigmas = np.zeros(numcodes)


    print "begin reading in data..."
    for i,xfile in enumerate(codes):
        direcpath = "%s%s"%(basepathpart,xfile)
        pointfile = "%s_point_estimates.out"%(labels[i])
        fullpath = os.path.join(direcpath,pointfile)
        #print fullpath
        data = np.loadtxt(fullpath)
        sz[i,:] = data[:,1]
        imag[i,:] = data[:,2]
        zpeak[i,:] = data[:,3]
        zweight[i,:] = data[:,4]
        print "read in data for %s"%xfile
        

#set up parameters
    imagcut = 50.0 #already have i<25.3 in the selection of gold sample
    cutsig = 0.02
    binedges = np.arange(0.0,2.01,0.02)
    centbins = 0.5*(binedges[1:] + binedges[0:-1])
    xbins,ybins = np.meshgrid(centbins,centbins)
    xspan = [[0.0,2.0],[0.0,2.0]]

    #numcodes = 11
    numrows = 3
    numcols = 4

#make plot

    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12.,9.))
    #fig.subplots_adjust(hspace=0.0)
    #fig.subplots_adjust(wspace=0.0)
    fig.subplots_adjust(hspace=0.025, wspace=0.055)

    #contourlevels = [1.0,2.5,4.0,5.5,7.0,8.5,10.0]
    contourlevels3 = [0.0,0.05,0.075,0.15,1.0,3.0,7.0,9.0]
    for i in range(numcodes):

#first, calculate the sigmas for each code
        szpzobj = pmet.EvaluatePointStats(zpeak[i],sz[i],imag[i],
                                          imagcut=imagcut)
        tmpsig,tmpsiggold = szpzobj.CalculateSigmaIQR()
        print("sigma for code %s is %g\n"%(labels[i],tmpsiggold))
        outfp.write("sigma for code %s is %g\n"%(labels[i],tmpsiggold))
        biasall,biasgold = szpzobj.CalculateBias()
        print "bias for code %s is %g\n"%(labels[i],biasgold)
        outfp.write("bias for code %s is %g\n"%(labels[i],biasgold))
        outlierall,outliergold = szpzobj.CalculateOutlierRate()
        print "Cat outlier frac for code %s is %g\n"%(labels[i],outliergold)
        outfp.write("Cat outlier frac for code %s is %g\n"%(labels[i],
                                                            outliergold))


        sigmas[i] = tmpsiggold
#        tmplabel = "%s $\sigma$=%.3f"%(labels[i],tmpsiggold)
        tmplabel = "%s"%(labels[i])
        ax = plt.subplot(numrows,numcols,i+1)
        ax.scatter(szpzobj.sz_magcut,szpzobj.pz_magcut,marker='.',s=10,c='b',
                   edgecolor='none',lw=1,label=tmplabel,alpha=0.2)
        pal = sbn.light_palette("navy", as_cmap=True)
        sbn.kdeplot(szpzobj.sz_magcut,szpzobj.pz_magcut,shade=True,
                    shade_lowest=False ,levels=contourlevels3,
                    gridsize=100,cmap=pal)
        ###sbn.kdeplot(szpzobj.sz_magcut,szpzobj.pz_magcut,shade=True,shade_lowest=False,gridsize=20,n_levels=15, cmap="Reds",cbar=True)


        outliersig = np.maximum(3.0*sigmas[i],0.06)

        upper_l = outliersig
        upper_h = 2.0 + 3.0*(outliersig)
        lower_l = -1.*outliersig
        lower_h = 2.0 - 3.0*(outliersig)
        ax.plot([0.,2.],[upper_l,upper_h],c='r',lw=2,linestyle='--')
        ax.plot([0.,2.],[lower_l,lower_h],c='r',lw=2,linestyle='--')
        ax.plot([0.,2.],[0.,2.],c='k',lw=2)

        plt.yticks(np.arange(0.,1.99,0.5),fontsize=14)
        plt.xticks(fontsize=14)
        
        ax.set_xlim([0.0,1.99])
        ax.set_ylim([0.0,1.99])
        ax.legend(loc="upper left",scatterpoints=1,fontsize=16)
        if i<(numcodes-4):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("${z_{spec}}$",fontsize=24)
        if i%4==0:
            ax.set_ylabel("${z_{phot}}$",fontsize=24)
        else:
            ax.set_yticklabels([])

    currenttime = time.time()
    print "took %g seconds"%(currenttime-starttime)
    plt.savefig("ZPEAK_szpz_threecolumn_12codes_biglabels.jpg",format='jpg')
    #plt.savefig("szpz_fourcolumn_11codes_navy.pdf",format='pdf')
    #plt.show()

    outfp.close()
    print ("finished")
if __name__=="__main__":
    main()
