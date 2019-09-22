from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np
import math
import pandas as pd
from operator import itemgetter

# First import the data....
# I changed 'NA' to -1



#---------------------Functions-------------------------------------------------
#get number of columns in tab delimited file

def getNumCol(file):
    with open(file, 'r') as f:
        num_cols = len(f.readline().split())
        f.seek(0)
    return(num_cols)


#outputs arr of length of exArr with the pearson coefficients
# of each index compared to sample.
def pCorr(sample,sampleIndex,exArr,clArr):


    #compare each index of the array with pearson correlation.
    #append the correlation to a new 1D array with respective indexes

    pcArr = [None] * len(exArr)

    i = 0
    while (i < len(exArr)):
        # print('PC')
        pC = np.corrcoef(sample, exArr[i])[0,1]#calculate pearson coefficient
        # print(pC)
        pcArr[i] = pC
        i = i + 1


    # print (pcArr) #check
    return(pcArr)


#outputs a tuple with sensativity level out of k to drug:
#                                        (drug, sensativity out of K)
def classify(sample,sampleIndex,exArr,clArr,sensArr,medArr,k):

    # print('RUNNING KNN CLASSIFIER:')
    pC = pCorr(sample,sampleIndex,exArr,clArr) #pearson correlation


    #take the pearson correlation, match it to cell line names,
    #then order from highest to lowest and take the first k.
    data = list(zip(clArr,pC,sensArr))
    data.sort(key=itemgetter(1))
    data.reverse() #decending order

    # print('DATA:')
    # print(data[0])
    # print(data[1])
    # print(data[2])
    # print(data[3])
    # print(len(medArr))
    # print(len(data))

    #sum which drugs top k are sensative too.

    #-------------------Return array of all the data-------

    return(data)


    #-------------------Show Medicine data--------------------#

    # i = 0
    # j=0

    # #medicine in array
    # medSum = [[0,0],[0,0],[0,0],[0,0],[0,0]]  #sensitivity score sum
    #
    #
    # while(i < k):
    #     while(j < len(medArr)):
    #         if(data[i][2][j] == 1):
    #             medSum[j][0] += 1
    #
    #         if(data[i][2][j] !=-1):
    #             medSum[j][1] += 1
    #
    #         j+=1
    #
    #     j = 0
    #     i+=1
    #
    #
    #
    # result = list(zip(medArr,medSum))
    # # print (result)
    # #print sensativity to which drugs
    # i = 0
    #
    # print(result[0])
    #
    # print('\nInputted Sample sensitivity results: ')
    # while(i < len(medSum)):
    #     print(' Drug: ', medArr[i] , '|' , 'Sensitivty: (' , medSum[i][0] , '/', k , ') = ' , medSum[i][0]/k)
    #     i+=1
    #
    # print('\nThe sample is sensative to the following drugs: ')
    # i = 0
    # while(i < len(medSum)):
    #     if (medSum[i][0] > 0):
    #         print('', medArr[i])
    #     i+=1
    # print('\n')
    #
    # return result



#removes  multiple indexs of test sample from array if need be
#returns new array to use.
            #array
def rmvInd(sampleIndex,arr,trans = False):

    newArr = np.copy(arr, order='K')
    if(trans == True):
        newArr = np.transpose(newArr)

    #convert to pandas dataframe to make sure everythin is dropped.
    df = pd.DataFrame(newArr)
    df = df.drop(sampleIndex)

    newArr = df.to_numpy()

    # print(newArr[sampleIndex])
    # print(arr[sampleIndex])
    return newArr


#Gives you all the data for a specific medicine index and removes all with -1 (unknowns)
#input k = 0 to do whole array, otherwise top k will be taken
def classMed(data,med,medArr,k=0):

    i = 0

    while (i < len(medArr)):
        if(med == medArr[i]):

            medIndex = i
            # print('medIndex')
            # print(medIndex)
            break
        i+=1

    # ind 0 = name ; ind 1 = pcorr ; ind 2 = sens arr
    #
    medData = []
    i = 0
    j = 0

    while (i < len(data)):
         # check to see if the medicine is the one were looking for value in variable:
        if(data[i][2][medIndex] != -1):
            medData.append([data[i][0][0], data[i][1], data[i][2][medIndex]])

        i+=1
    return(medData)


def rmvUnknownMed(clArr,sensArr,med,medArr,n=5):

    #set as
    if (n == 0):
        n = len(medArr)
    i = 0

    while (i < n): # find index of specific med
        if(med == medArr[i]):
            medIndex = i
            break
        i+=1

    rmvInd = []
    i = 0

    while (i < len(clArr)):
        if(sensArr[i][medIndex] == -1):
            rmvInd.append(i)
        i+=1
    return(rmvInd)




def isSensitive(data,k,ratio):
    isSensitive = False
    i = 0
    sens = 0
    total = 0
    while(i<k):
        if(data[i][2] == 1):
            sens += 1
            total+= 1
        if(data[i][2] == 0):
            total+= 1

        i+=1
    #########return T/F ############
    # if((total/k) >= ratio):
    #     isSensitive = True

    # return(isSensitive)
    #########return ratio ############
    # return((sens/total))

    #########return tuple of ratio ######
    return((sens,total))



#gets the cell line data about one medicine and predictions
def getSens(exArr,clArr,sensArr,medArr,medIndex,k,threshold):
    returnData = []
    rmvMed = rmvUnknownMed(clArr,sensArr,medArr[medIndex],medArr,5)
    exArrB = rmvInd(rmvMed,exArr)
    clArrB = rmvInd(rmvMed,clArr)
    sensArrB = rmvInd(rmvMed,sensArr)

    i = 0
    while(i < len(clArrB)):
        #remove the index to be tested on in the array
        exArrA = rmvInd(i,exArrB)
        clArrA = rmvInd(i,clArrB)
        sensArrA = rmvInd(i,sensArrB)

        # rmvMed = rmvUnknownMed(clArrA,sensArrA,medArr[0],medArr,5)
        # exArrB = rmvInd(rmvMed,exArrA)
        # clArrB = rmvInd(rmvMed,clArrA)
        # sensArrB = rmvInd(rmvMed,sensArrA)


        data = classify(exArrB[i],i,exArrA,clArrA,sensArrA,medArr,k)
        medData = classMed(data,medArr[medIndex],medArr,k)
        s = isSensitive(medData,k,threshold)


        cName = clArrB[i][0]
        actualSens= sensArrB[i][medIndex]

        data2 = [cName,s,(s[0]/s[1]),actualSens]
        # print(data2)
        returnData.append(data2)

        i= i+1

    return(returnData)

def cValidate(exArr,clArr,sensArr,medArr,k,threshold = .5):

    data0 = getSens(exArr,clArr,sensArr,medArr,0,k,threshold)
    data1 = getSens(exArr,clArr,sensArr,medArr,1,k,threshold)
    data2 = getSens(exArr,clArr,sensArr,medArr,2,k,threshold)
    data3 = getSens(exArr,clArr,sensArr,medArr,3,k,threshold)
    data4 = getSens(exArr,clArr,sensArr,medArr,4,k,threshold)




    # print(data0)
    arr = [data0,data1,data2,data3,data4]

    return(arr)


def sensativity(TP,FN):
    return (TP / (TP + FN))


def specificity(TN,FP):
    return(TN / (TN + FP))

#sort the cell lines based on their most sensative neighbors to the medicine ratio
def sensSort(arr):

    sortedArr = sorted(arr, key=itemgetter(2))
    sortedArr.reverse()
    # print(sortedArr)
    return sortedArr
# compare sorted list to sensativity to certain cell
# sort score list

def threshold(arr,threshold,flag):
    # print('Threshold')
    # print(arr)

    if (flag == 1):
        dataIndex = 4  #choose special sort data
    else:
        dataIndex = 1

    i = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    while(i < len(arr)):
        if (flag == 1):
            nThresh = int(arr[i][4])  #for special scoring
        else:
            nThresh = arr[i][1][0] # number sensative in the tuple.

        # print('starting threshold')
        # print(nThresh)

        if(nThresh >= threshold):
            if (arr[i][3] == 1 ): #true positive
                tp +=1
            if (arr[i][3] == 0 ): #false positive
                fp +=1
        if(nThresh < threshold):
            if (arr[i][3] == 1 ): #true negative
                tn +=1
            if (arr[i][3] == 0 ): #false negative
                fn +=1
        i +=1

    # print('ending threshold')


    result = [tp,fp,fn,tn]
    print('[tp,fp,fn,tn]' , threshold)
    print(result)
    return(result)




# number of threshold will be = k
#will calcualte the sensativity/ specificty for each graph then print ROC curve
def computeROC(arr,k, title, medicine,threshMin = 1,threshMax = 1000,flag = 0):
    #arr [x] [0] = name, [1] = (number sensative,k) [2] = ratio, [3] = actual sensativity, [4] = specialScore


    #reset defaults / no inputs for part A

    if (threshMax == 1000):
        threshMax = k

    print('ROC curve')
    # arr = sensSort(arr)
    xData = []
    yData = []
    index = []
    i = threshMin

    while(i <= threshMax):

        result = threshold(arr,i,flag)
        result.append(['Threshold: ' + str(i)]) # all fp,tp,fn,tn
        # 0 = tp , 1 = fp , 2 = fn , 3 = tn
        #compute the specificity / sensativity and sets as coordinates
        #arrays are from 1 to 0
        x = specificity(result[3],result[1])
        x = 1 - x
        # print(x)
        y = sensativity(result[0],result[2])
        xData.append(x)
        yData.append(y)
        i+=1

    print(xData,yData)

    #for reference:
      #print('[tp,fp,fn,tn]')
      #sensativity(TP,FN):
            #return (TP / (TP + FN))
      #specificity(TN,FP):
#           #return(TN / (TN + FP))


    # printData = list(zip(result,xData,yData))
    # print()



    

    print('1-specificy',xData[0])
    print('sensitivity',yData[0])


    plt.plot(xData,yData,label = medicine)
    plt.title(title)
    plt.xlabel('1 - specificity')
    plt.ylabel('sensitivity')
    plt.legend()
    plt.plot([0,1], color = 'black' , linestyle = '--' )
    plt.show()

    return([xData],[yData])




#get pearson data based on array of arrays where cell line names are the 0 index of inner arrays..
def getPearsonData(arr,exArr,clArr,sensArr,medArr,k):

    #retrieve data set that is included in arr from orginal exArr/clArr

    # print(arr)
    # print('\n')


    i = 0
    j = 0
    newEx =[] # new array of expression data matching indexs of arr
    newCl =[]
    newSens = []

    while(i<len(arr)):

        while(j < len(clArr)):

            if(clArr[j] == arr[i][0]):
                # print(j)
                newEx.append(exArr[j])
                newCl.append(clArr[j])
                newSens.append(sensArr[j])

            j+=1
        i+=1
        j=0

    #pCData = classify(sample,sampleIndex,exArr,clArr,sensArr,medArr,k)
    i = 0
    pData = []
    allPData = []


    while(i < len(arr)):

        ExTemp = rmvInd(i,newEx)
        ClTemp = rmvInd(i,newCl)
        SensTemp = rmvInd(i,newSens)



        pData = classify(newEx[i],i,ExTemp,ClTemp,SensTemp,medArr,k)         # data about pearson correlations

        c = len(pData) - k

        #get rid of all except k
        while(c > 0):
            pData.pop()
            c -= 1

        allPData.append(pData)
        i+=1

    # print(allPData[0])
    #allPData has all the top K pearson correlations for the data inputted.
    #pdata [i][0] = name  , [i][1] = pCorr , [i][2] = sensativities
    return(allPData)



#calculate score based off inputted array index
def specialScore(arr,pData,index,medIndex,k):

    #score equation:
    i=0
    sign = 0
    score = 0
    # print(pData[0])

    while(i<len(pData[index])):

        sign = pData[index][i][2][medIndex]
        if(sign != 1):
            sign = -1

        s = sign * pData[index][i][1]
        score = score + s
        i+=1

    # print(pData[index][0][1])
    # print(score)
    return(score)








#sort based of special score, output new index that is sorted.
def specialSort(arr,pData,medIndex,k):
        #pdata
        #pdata [i][0] = name  , [i][1] = pCorr , [i][2] = sensativities
        #arr [i][0] = name  , [i][1] = tuple of sesativity , [i][2] = ratio, [i][3] = actual
        #for each item in array, get score:

        # print(len(arr))

        scoreList = []
        indexList = []
        i=0

        newArr = arr.copy()

        while(i<len(arr)):
            score = specialScore(arr,pData,i,medIndex,k)
            scoreList.append(score)
            newArr[i].append(score)
            i+=1


        # toBeSorted = list(zip(arr,scoreList))
        # sortedArr = sorted(toBeSorted, key=itemgetter(1))
        # sortedArr.reverse()
        newArr = sorted(newArr, key=itemgetter(4))
        newArr.reverse()

        # print(sortedArr)
        # print('\n')
        # print(newArr)
        #['MCF7', (4, 5), 0.8, 1.0, 2.810801379070652]
        return(newArr)



# find the smallest special score in the array
def findMinMaxInteger(arr):

    i = 0
    max = arr[i][4]
    min = arr[i][4]

    while(i < len(arr)):
        if (arr[i][4] >= max):
            max = arr[i][4]

        if (arr[i][4] <= min):
            min = arr[i][4]
        i+=1

    min = int(min)
    max = int(max)

    # print(min,' ',max)
    return(min,max)

def multiplyScores(arr,x):

    newArr = arr.copy()
    i = 0

    while(i < len(newArr)):
        newArr[i][4] = (newArr[i][4]) * x
        i+=1

    # print(newArr)
    return(newArr)






#-------------------Notes-------------------
#Modification I did to 'DREAM_data.txt':
#Converted the N/A to -1's for the sake of putting sensitivity into numpy arrays





#_________________Main Functions________________________



#
#Pulling data from 'DREAM_data.txt' file :
numCols = getNumCol('DREAM_data.txt') #total number of columns in tab delimited file
clArr = np.loadtxt('DREAM_data.txt', dtype = str, delimiter='\t',max_rows =1 ,usecols =range(1,numCols), unpack = True ) #cell line names (46 of them)
exArr = np.loadtxt('DREAM_data.txt', delimiter='\t',skiprows = 6 ,usecols =range(1,numCols), unpack = True ) #gene expression data (below medicine sens.)
sensArr = np.loadtxt('DREAM_data.txt', delimiter='\t',skiprows = 1, max_rows = 5 ,usecols =range(1,numCols), unpack = True ) #array of medicine sensitivity
medArr = np.loadtxt('DREAM_data.txt', dtype = str,delimiter='\t',skiprows = 1, max_rows = 5 ,usecols = range(0,1), unpack = True ) #names of medicine

#print(medArr)
# print(exArr[0])
# pCorr(exArr[0],exArr,clArr)

#return which medicine the sample is sensetive too, uses K nearest neighbors approach.
#input the drug to classify for as well.

k = 5
sample = 1

# #---- Med Data -----
# # Everolimus(mTOR) = 0
# # Disulfiram(ALDH2) = 1
# # Methylglyoxol(Pyruvate) = 2
# # Mebendazole(Tubulin) = 3
# # 4-HC(D-1 alkylator) = 4


#PROBLEM #2
arr =cValidate(exArr,clArr,sensArr,medArr,k,.5)
print(arr[0])
# print('0000000000000000000000000000000')
# computeROC(sensSort(arr[0]),k, str(medArr[0]) +  'KNN (k=' + str(k) + ') ROC Curve & Random Classification',str(medArr[0]))
# print('1111111111111111111111111111111')
# computeROC(sensSort(arr[1]),k, str(medArr[1]) +  'KNN (k=' + str(k) + ') ROC Curve & Random Classification',str(medArr[1]))
# print('2222222222222222222222222222222')
computeROC(sensSort(arr[2]),k, str(medArr[2]) +  'KNN (k=' + str(k) + ') ROC Curve & Random Classification',str(medArr[2]))
# print('3333333333333333333333333333333')
# computeROC(sensSort(arr[3]),k, str(medArr[3]) +  'KNN (k=' + str(k) + ') ROC Curve & Random Classification',str(medArr[3]))
# print('4444444444444444444444444444444')
# computeROC(sensSort(arr[4]),k, str(medArr[4]) +  'KNN (k=' + str(k) + ') ROC Curve & Random Classification',str(medArr[4]))

#Problem #3
#A
k=5
#
arr =cValidate(exArr,clArr,sensArr,medArr,k,.5)
#
# computeROC(sensSort(arr[0]),k, 'K = ' + str(k) ,str(medArr[0]))
# computeROC(sensSort(arr[1]),k, 'K = ' + str(k) ,str(medArr[1]))
# computeROC(sensSort(arr[2]),k, 'K = ' + str(k) ,str(medArr[2]))
# computeROC(sensSort(arr[3]),k, 'K = ' + str(k) ,str(medArr[3]))
# computeROC(sensSort(arr[4]),k, 'K = ' + str(k) ,str(medArr[4]))
# #
# plt.show()


#B

pData = getPearsonData(arr[0],exArr,clArr,sensArr,medArr,k)
arr[0] = specialSort(arr[0],pData,0,k)

# print(arr[0])
# i = 0
# while(i < len(arr[0])):
#     print(i)
#     print(i,'cell line: ', arr[0][i][0],' score:',arr[0][i][4])
#     i+=1





# arr[0] = multiplyScores(arr[0],40)
# min,max = findMinMaxInteger(arr[0])
# computeROC(sensSort(arr[0]),k, str(medArr[0]) + '; Special sort K = ' + str(k) ,str(medArr[0]),min,max,1)
# plt.show()
#
# pData = getPearsonData(arr[1],exArr,clArr,sensArr,medArr,k)
# arr[1] = specialSort(arr[1],pData,1,k)
# arr[1] = multiplyScores(arr[1],40)
# min,max = findMinMaxInteger(arr[1])
# computeROC(sensSort(arr[1]),k, str(medArr[1]) + 'Special sort K = ' + str(k) ,str(medArr[1]),min,max,1)
# plt.show()
#
#
# pData = getPearsonData(arr[2],exArr,clArr,sensArr,medArr,k)
# arr[2] = specialSort(arr[2],pData,2,k)
# arr[2] = multiplyScores(arr[2],40)
# min,max = findMinMaxInteger(arr[2])
# computeROC(sensSort(arr[2]),k, str(medArr[2]) + 'Special sort K = ' + str(k) ,str(medArr[0]),min,max,1)
# plt.show()
#
# pData = getPearsonData(arr[3],exArr,clArr,sensArr,medArr,k)
# arr[3] = specialSort(arr[3],pData,3,k)
# arr[3] = multiplyScores(arr[3],40)
# min,max = findMinMaxInteger(arr[3])
# computeROC(sensSort(arr[3]),k, str(medArr[3]) + 'Special sort K = ' + str(k) ,str(medArr[0]),min,max,1)
# plt.show()
#
# pData = getPearsonData(arr[4],exArr,clArr,sensArr,medArr,k)
# arr[4] = specialSort(arr[4],pData,4,k)
# arr[4] = multiplyScores(arr[4],40)
# min,max = findMinMaxInteger(arr[4])
# computeROC(sensSort(arr[4]),k, str(medArr[4]) + 'Special sort K = ' + str(k) ,str(medArr[0]),min,max,1)
# plt.show()
#
#
















print('____________________________________________________________________________________________')
