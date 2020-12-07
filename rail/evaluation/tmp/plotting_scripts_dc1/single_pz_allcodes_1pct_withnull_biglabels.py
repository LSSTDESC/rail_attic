#Script to make a plot of p(z) for Nth galaxy (read from command line) for
#all codes in the DC1 data challenge
#Jan 11, 2018
#Sam Schmidt
import sys,os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sps
import string
#import seaborn as sns


def getunsortedwhichline(direcpath,whichline):
    unsortfile = "goldunorderedidline.out"
    totalpath = os.path.join(direcpath,unsortfile)
    print("total path: %s"%totalpath)
    tmpdata = np.loadtxt(os.path.join(direcpath,unsortfile))
    tmpidd = np.int32(tmpdata[:,0])
    strid = np.array([str(np.int32(i)) for i in tmpidd])
    linenum = np.int32(tmpdata[:,1])
    iddict = dict(zip(strid,linenum))
    finid = np.int32(iddict[np.str(whichline)])
    print ("found line %d for ID %d"%(finid,whichline))
    return finid

def main(argv):
    if len(argv) != 2:
        print( "usage: plotstuff.py [line number of pz]\n")
        exit(2)
    whichline = np.int32(argv[1])
    totlines = 3993 #size of gold sample files
    bigtotlines = 399356
    endskiplines = totlines - whichline

    basepathpart = "./TESTDC1"
    codes = ("ANNZ2","BPZ","DELIGHT","EAZY","FLEXZ","GPZ/AUG2018","LEPHARE",
             "METAPHOR","NN","SKYNET","TPZ","NULL2")
    labels = ("ANNz2","BPZ","Delight","EAZY","FlexZBoost","GPz","LePhare",
              "METAPhoR","CMNN","SkyNet","TPZ","TrainZ")
    #files = ("1pct_ANNz2gold","1pct_BPZgold", "1pct_Delightgold_v3", "1pct_EAZYgold", "1pct_Dec1Flexzgold", "1pct_GPZgold","1pct_LEPHAREgold", "1pct_Metaphorgold", "dec8NNgold", "Skynetgold", "TPZgold")
    files = ("1pct_ANNz2gold","1pct_BPZgold", "1pct_Delightgold", 
             "1pct_EAZYgold", "1pct_Mar5Flexzgold", "1pct_GPZgold",
             "1pct_LEPHAREgold", "1pct_Metaphorgold", "dec8NNgold", 
             "Skynetgold", "TPZgold","1pct_null")

#    filesorted = (True,True,True,True,True,True,True,True,False,False,False)
    filesorted = (True,True,True,True,True,True,True,True,False,False,False,
                  True)

    labeldict = dict(zip(codes,labels))
    filedict = dict(zip(codes,files))
    
    szfile = "TESTDC1BPZ/1pct_BPZgold_idszmag.txt"
    tmpdata = np.loadtxt(szfile)
    truez = tmpdata[whichline-1,1]
    whichtrueid = np.int32(tmpdata[whichline-1,0])
    truemag = tmpdata[whichline-1,2]

    print ("grabbing galaxy from line %d of file\n"%whichline)

    numcodes = len(codes)
    numrows = 3
    numcols = 4

    #fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(16,12))
    #fig.subplots_adjust(hspace=0.0)
    #fig.subplots_adjust(wspace=0.0)
    

    fig,axes = plt.subplots(numcodes, sharex=True, sharey=True, figsize=(12.1,9))
    fig.subplots_adjust(hspace=0.025, wspace=0.055)
    #sns.set()

    absmax = -9999.

    for i,xfile in enumerate(codes):
        xfilename = files[i]+"_pz.out"
        direcpath = "%s%s"%(basepathpart,xfile)
        fullpath = os.path.join(direcpath,xfilename)
        #print fullpath
        print ("opening file %s..."%fullpath)
            
        if filesorted[i]:
            grabline = whichline
            tmpendskip = endskiplines
        else:
            grabline = getunsortedwhichline(direcpath,whichtrueid)
            tmpendskip = bigtotlines-grabline

        pzdata = np.genfromtxt(fullpath,skip_header=grabline-1,skip_footer=tmpendskip)
        print ("pzdata:")
        print (pzdata)
        zpath = os.path.join(direcpath,"zarrayfile.out")
        print ("opening file %s..."%zpath)
        zarray = np.loadtxt(zpath)
        print (pzdata.shape)
        print (zarray.shape)
            
        #normalize pzdata:
        #delgrid = zarray[1:] - zarray[:-1]
        #xgrid = np.append(delgrid,delgrid[-1])
        #normsum = np.sum(pzdata*xgrid)
        normy = np.amax(pzdata)
        pzdata /= normy
#        maxval = np.amax(pzdata)*1.35
#        if maxval > absmax:
#            absmax = maxval
        maxval = 1.35

        print ("read in data for %s"%xfile)
        
#        #Make qp.PDF object
#        qpobj = qp.PDF(gridded=(zarray,pzdata))
#        newarray = np.linspace(0.,2.,1000)
#        qpres = qpobj.evaluate(loc=newarray,using='gridded',norm=False)
#        xxx = qpres[0]
#        yyy = qpres[1]
                       


#PLOTTING STUFF
    
        ax = plt.subplot(numrows,numcols,i+1)
        tmplabel = "%s"%(labels[i])
        plt.plot(zarray,pzdata,c='b',linestyle='-',linewidth=3,label=tmplabel)
#take out qp
#        plt.plot(xxx,yyy,c='r',linestyle='-',linewidth=1,label="qp fit")
        plt.plot([truez,truez],[0.,maxval],c='r',linestyle='-',linewidth=2)
        ax.set_xlim([0.0,1.99])
        ax.set_ylim([0.0,maxval])
        ax.legend(loc="upper left",fontsize=16, bbox_to_anchor=(0.0,1.))
        if (i<(numcodes-4)):
            ax.set_xticklabels([])
        else:
            #ax.set_xticklabels(['',0.2, 0.4, 0.6, 0.8, 1.0,1.2,1.4,1.6,1.8,2.0])
            ax.set_xlabel("redshift",fontsize=22)
        if i%4==0:
            ax.set_ylabel("p(z)",fontsize=22)
            #if i>0:
            fig.canvas.draw()
            #y_labels = [item.get_text() for item in ax.get_yticklabels()]
            #y_labels[-1] = ''
            #ax.set_yticklabels(y_labels)
            ax.set_yticklabels([])
            #ax.yaxis.set_ticks_position('none')

        else:
            ax.set_yticklabels([])
            #ax.yaxis.set_ticks_position('none')

        plt.xticks(fontsize=16)
        #plt.yticks(fontsize=12)

   # tmptitle = "Buzzid: %d mag: %s specz: %s"%(whichtrueid,truemag,truez)
   # plt.suptitle(tmptitle,fontsize=22)
    print ("buzzID of gal: %d"%whichtrueid)
    outputname = "pz_12codes_%d_biglabels.jpg"%whichtrueid

    print ("absmax to set plot to: %g\n"%absmax)

    plt.savefig(outputname,format='jpg')
#    plt.show()

if __name__=="__main__":
    main(sys.argv)
