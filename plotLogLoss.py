#!/usr/bin/python
import sys, os
import matplotlib.pyplot as plt


def plotLogLoss(logName):
    f = open(logName, 'r')
    numList = []

    AveIterNum = 100
    iterCount = 0
    numAcc = 0
    
    
    for line in f:
        if str.find(line, 'loss') != -1 and \
            str.find(line, 'Iter') != -1:
            numStr = line[str.find(line,'=')+1:str.find(line,'(')]

            numAcc += float(numStr)
            iterCount += 1

            if iterCount % AveIterNum == 0:
                print(iterCount, numAcc)
                numList.append(numAcc / iterCount)
                iterCount = 0
                numAcc = 0

    plt.plot(numList)
    plt.show()


def getLogName(argv):
    if (len(argv) <= 1):
        return None
    if not os.path.exists(argv[1]):
        return None
    return argv[1]

if __name__ == '__main__':
    logName = getLogName(sys.argv)
    if (logName != None):
        plotLogLoss(logName)
    else:
        print('please input filename or input correct filename')
