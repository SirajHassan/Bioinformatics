

#Written by Siraj Hassan
#CSCI 5461 HW 1
######################################

from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import math
import pandas as pd


#take in numpy array



def quantileNormalize(df):
    #https://stackoverflow.com/questions/37935920/quantile-normalization-on-pandas-dataframe
    #written by user: ayhan
    df
    rank_mean = df.stack().groupby(df.rank(method='first').stack().astype(int)).mean()
    ndf = df.rank(method='min').stack().astype(int).map(rank_mean).unstack()
    ndf.head()
    return ndf

#---------------------------------------------


#---------------------------------------------

# writes out Log 10 applied to wang_data to be viewed in excel as csv
def writeOutLog():
    # I COULD MAKE A FUNCTION FOR THIS BUT IM NOT TRYING TO BE PRETTY Right now
    with open('wang_data.txt', 'r') as f:
        num_cols = len(f.readline().split())
        f.seek(0)

    Data = np.loadtxt('wang_data.txt', delimiter='\t',skiprows = 2 ,usecols = range(2,num_cols))
    logData = np.log10(Data)

    df = pd.DataFrame(logData)
    df.to_csv('logData.csv',index=False)



# writes out normalized log(10) data
def writeOutNorm():

    with open('wang_data.txt', 'r') as f:
        num_cols = len(f.readline().split())
        f.seek(0)

    Data = np.loadtxt('wang_data.txt', delimiter='\t',skiprows = 2 ,usecols = range(2,num_cols))
    logData = np.log10(Data)
    df = pd.DataFrame(logData)
    ndf = quantileNormalize(df)
    ndf.to_csv('normData2.csv',index=False)

#make new files
#writeOutNorm()
# writeOutLog()



#load file
# get rid of the first 2 columns
#find number of columns  - source: https://stackoverflow.com/questions/13311471/skip-a-specified-number-of-columns-with-numpy-genfromtxt
with open('wang_data.txt', 'r') as f:
    num_cols = len(f.readline().split())
    f.seek(0)

# having issues with the txt file so I converted wang to a csv
#print (num_cols)


#--------------------------------------------------------------------------
def threeA():

    #nData is per row
    aData = np.loadtxt('wang_data.txt', delimiter='\t',skiprows = 2 ,usecols = range(2,num_cols))

    #log transform
    # aData = np.log10(aData)

    data = aData.tolist()
    #print (data[0])

    flatList = []
    for sublist in data:
        for val in sublist:
            flatList.append(val)

    print (flatList[0])

    #normal
    plt.hist(flatList, bins=200, range = (0,2000))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('3a Normal')
    plt.show()

    #log transform

    # print (flatList[0])
    #
    # plt.hist(flatList, bins=200, range = (0,max(flatList)))
    # plt.xlabel('expression levels')
    # plt.ylabel('probe counts')
    # plt.title('3a logTransformed' )
    # plt.show()

    # length = len(flatList)
    # print(length)
    # print(flatList[length-1])

threeA()





#-------------------------------------------------------------------------------
def threeB():

    #nData is per row for part B we want per column (first 4)

    #first 4 columns
    #unpack = True will transpose
    bData = np.loadtxt('wang_data.txt', delimiter='\t',skiprows = 2 ,usecols = range(2,6) , unpack = True)
    logBData = np.log10(bData)



    #bData contains the first 4 columns..
    # print (bData[0])
    # print (logBData[0])

    #4 columns now in list and logbased
    logCol1 = logBData[0].tolist()
    logCol2 = logBData[1].tolist()
    logCol3 = logBData[2].tolist()
    logCol4 = logBData[3].tolist()


    #make histogram for each (they seem pretty similair):
    plt.hist(logCol1, bins=200, range = (0,max(logCol1)))

    print(np.mean(logCol1))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Log Base 10 Data Sample 1' )
    plt.show()
    plt.hist(logCol2, bins=200, range = (0,max(logCol2)))

    print(np.mean(logCol2))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Log Base 10 Data Sample 2' )
    plt.show()
    plt.hist(logCol3, bins=200, range = (0,max(logCol3)))

    print(np.mean(logCol3))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Log Base 10 Data Sample 3' )
    plt.show()
    plt.hist(logCol4, bins=200, range = (0,max(logCol4)))

    print(np.mean(logCol4))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Log Base 10 Data Sample 4' )
    plt.show()

    plt.hist(logCol1, bins=200, range = (0,max(logCol1)))
    plt.hist(logCol2, bins=200, range = (0,max(logCol2)))
    plt.hist(logCol3, bins=200, range = (0,max(logCol3)))
    plt.hist(logCol4, bins=200, range = (0,max(logCol4)))
    plt.show()

# threeB()
#--------------------------------------------------------------------------------------------------------


def threeC():

    #for this I need to convert the numpy array into a pandas dataframe and use some code I
    # found on github.
    # https://github.com/ShawnLYU/Quantile_Normalize
    # https://stackoverflow.com/questions/50624046/convert-numpy-array-to-pandas-dataframe





    # Data = np.loadtxt('wang_data.txt', delimiter='\t',skiprows = 2 ,usecols = range(2,num_cols), unpack = True )
    # logData = np.log10(Data)
    # ldf = pd.DataFrame(logData)
    #
    #
    # # now we have the df, lets normalize it
    # df = quantileNormalize(ldf)
    #
    #
    # # print(type(df))
    # nArr = df.values

    nArr = np.loadtxt('normData2.txt', delimiter='\t', unpack = True)

    print(nArr[0])
    # print(nArr[1])

    Col1 = nArr[0].tolist()
    Col2 = nArr[1].tolist()
    Col3 = nArr[2].tolist()
    Col4 = nArr[3].tolist()

    print(np.mean(Col1))
    plt.hist(Col1, bins=200, range = (0,max(Col1)))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Quantile Nomarlization Data Sample 1' )
    plt.show()
    print(np.mean(Col2))
    plt.hist(Col2, bins=200, range = (0,max(Col2)))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Quantile Nomarlization Data Sample 2' )
    plt.show()
    print(np.mean(Col3))
    plt.hist(Col3, bins=200, range = (0,max(Col3)))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Quantile Nomarlization Data Sample 3' )
    plt.show()
    print(np.mean(Col4))
    plt.hist(Col4, bins=200, range = (0,max(Col4)))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Quantile Nomarlization Data Sample 4' )
    plt.show()

    plt.hist(Col1, bins=200, range = (0,max(Col1)))
    plt.hist(Col2, bins=200, range = (0,max(Col2)))
    plt.hist(Col3, bins=200, range = (0,max(Col3)))
    plt.hist(Col4, bins=200, range = (0,max(Col4)))
    plt.xlabel('expression levels')
    plt.ylabel('probe counts')
    plt.title('Quantile Nomarlization All Four Samples' )
    plt.show()





# threeC()


#--------------------------------------------------------------------------------------------------------

def four(flag):   #flag = 0 -> t test, #flag = 1 -> ranksum ... function returns n array of the probes and their p values (acending order)
    # t test
    # take data, make 2 new sets of 22k rows
    #go through each row of original. If there is a 1 append to set same for 0
    #compare data and run T test

    #Norm data needs

    test = np.loadtxt('testResult.txt', delimiter='\t')
    # print(test[1])
    #test is a row of the test results
    normData = np.loadtxt('normData2.txt', delimiter='\t', skiprows = 0 )
    # print(normData[0])


    testList = test.tolist()
    normList = normData.tolist()

    # print(normList[0][0])
    # print(testList[0])


    zeroList = []
    oneList = []


    i = 0
    j = 0
    while (i < len(normList)) :
        tempListZero = []
        tempListOne = []
        while (j < len(testList)) :
            if (testList[j] == 0.0):
                tempListZero.append(normList[i][j])
            else:
                tempListOne.append(normList[i][j])
            j+= 1

        zeroList.append(tempListZero)
        oneList.append(tempListOne)
        j = 0
        i+= 1
        # print('outerloop')

    #now we a list of zeros and one expression levels for each probe
    # print (zeroList[(len(zeroList)-1)])
    # print (zeroList[(len(oneList)-1)])

    #time to do T test

    from scipy import stats

    #vector for t and p values comparing the probes
    tpVector = []
    tempStats = []
    #tVector = stats.ttest_ind( zeroList[0], oneList[0], equal_var = False)
    #print (tVector)
    length = len(zeroList)
    length = len(oneList)
    i = 0



    while(i < length):
        #choose t test of wilcox rank sum by commenting out
        if(flag == 0):
            tempStats = stats.ttest_ind( zeroList[i], oneList[i], equal_var = False)        # t test
        else:
            tempStats = stats.ranksums(zeroList[i], oneList[i])
            # tempStats = stats.mannwhitneyu( zeroList[i], oneList[i], use_continuity=False , alternative = 'two-sided' ) #old rank sum



                                   #wilcox rank sum
        tpVector.append(tempStats)
        i += 1
    #print(len(tpVector))

    geneList = np.loadtxt('wang_data.txt', dtype = np.str, delimiter='\t',skiprows = 2 ,usecols = range(0,2))
    #now we have list of all the genes
    # print(geneList[0][0])
    # print(len(geneList))


    tTest = []
    sig = 0
    i = 0

    #set alpha cutoff, .05 for problem 4; (.05 / 22283) = 0.00000224386 for problem 5 with Bonferroni correction
    cutoff = .05 #for problem 4



    while(i < length):
        # only append to list if the p value is below .05 (significant enough)
        if (tpVector[i][1] < cutoff):
            sig+=1
        tTest.append([geneList[i][0],geneList[i][1],tpVector[i][0],tpVector[i][1]])
                                                        #t stat or sum of rank  , p value
        i += 1

    #print the number of probes that are significant:

    print('number of significant probes : ' + str(sig))

        # now we have the genes and their t test results

    from operator import itemgetter    # sort list based on smallest p value

    # sort based on p value (decending right now, but should be acending if possible)
    #sorted(tTest, key=itemgetter(3))


    tTest.sort(key=itemgetter(3), reverse=False)

# Get all the p values into a list
    nlogPVal = []
    count = 0

    while(count < length):
        nlogPVal.append(-1 * math.log((tTest[count][3]),10))  # -log10 for 4b
        count +=1


    # def fourB(pv):
    #     # 4B
    #     print('4B')
    #     # make negative log histograms
    #     print(len(pv))
    #     plt.hist(pv, bins=200, range = (0,max(pv)))
    #     plt.xlabel('-log10(p value)')
    #     plt.ylabel('probe counts')
    #     plt.title('Rank Sum' )
    #     plt.show()
    #
    #
    #
    # fourB(nlogPVal)

# PRINT TOP 10 GENES / lowest 10 p Values
    # i = 0
    # while(i<10):
    #
    #     print (str(i+1) + ': ' + str(tTest[i]))
    #     i += 1





# NOTE: fiveA() and fiveB() need the stats.mannwhitneyu test to be used

    def fiveA():
        # Here well take the rank sum data, and set the cutoff at .05 / Number of probes (5003)
        bTest = tTest.copy()
        bTest.sort(key=itemgetter(3), reverse=False)
        N = len(tTest)
        print(N)
        alpha = .05
        bCutoff = alpha / N

        # i = 0
        # while(i < len(bTest)):
        #     #modify p values
        #     bTest[i][3] = bTest[i][3]/N
        #     i+=1

        bVec = []
        i = 0
        count2 = 0;
        while(i < len(bTest)):
            #modify p values
            if (bTest[i][3]< bCutoff):
                count2 += 1
            bVec.append(bTest[i][3])
            i+=1

        print('FiveA:' + str(count2))
        return(bVec)
        #bvec has the p val for the Bonferroni





    def fiveB():
        # Here well take the rank sum data, sort it by p value in ascending order (already sorted)
        # Then make a new vector with the p values modified: p = (p/N)*index

        N = len(tTest)
        print('N: ' + str(N))
        alpha = 0.05

        pBhVec = []
        pVec = fiveA()
        print (type(pVec))
        i = 0
        while(i < len(tTest)):
            #modify p values
            # pBh = ((tTest[i][3] * N) / (i+1))
            pBh = ((pVec[i] * (len(pVec))) / (i+1))
            pBhVec.append(pBh)
            i+=1
        print (type(pBhVec))
        print (type(tTest))

        print (pBhVec[0])
        print (pVec[0])
        print('pBhVec len :' + str(len(pBhVec)))


        count = 0
        j = 0
        #compare BH p value to threshold alpha, count significant
        while(j < len(tTest)):
            if( pBhVec[j] < alpha):
                count+=1
            j +=1

        print('BH count: ' + str(count))

        plt.plot(pBhVec[0:500], label = 'pBH')
        plt.plot(pVec[0:500] , label = 'pRankSum')
        plt.xlabel('index')
        plt.ylabel('p value')
        plt.legend()
        plt.show()

        # plt.plot(pBhVec[0:10000], label = 'pBH')
        # plt.plot(pVec[0:10000] , label = 'pRankSum')
        # plt.xlabel('index')
        # plt.ylabel('p value')
        # plt.legend()
        # plt.show()
        #
        # plt.plot(pBhVec[0:25000], label = 'pBH')
        # plt.plot(pVec[0:25000] , label = 'pRankSum')
        # plt.xlabel('index')
        # plt.ylabel('p value')
        # plt.legend()
        # plt.show()




        #make indices vs p value graph for the BH pvalue and the rank sum p value
    # fiveB()

    return(tTest)

four(1)   # 1 for rank sum 0 for t testList


def fourC(tTest,rSum):
    i =0
    j=0
    count = 0
    rVec = []
    tVec=[]

    while(i < len(tTest)):
        #modify p values
        if (tTest[i][3] < .05):  # check if significant
            #check if same gene is also in rSum
            while(j < len(rSum)):
                if (tTest[i][0] == rSum[j][0]):
                    count +=1
                    if(count <= 10):
                        rVec.append(rSum[j])
                        tVec.append(tTest[i])

                j+=1
        i+=1
        j=0


    k = 0
    m=0

    while(k< 10):
        print(rVec[k][0])
        k+=1
    k = 0

    while(k< 10):
        print(rVec[k][1])
        k+=1
    k = 0

    while(k< 10):
        print(rVec[k][3])
        k+=1
    k = 0


    print("....")

    while(m < 10):
        print(tVec[m][0])
        m+=1
    m = 0

    while(m < 10):
        print(tVec[m][1])
        m+=1
    m = 0

    while(m < 10):
        print(tVec[m][3])
        m+=1
    m = 0








    print('Overlapping Count:')
    print(count)


# fourC(four(0),four(1))
