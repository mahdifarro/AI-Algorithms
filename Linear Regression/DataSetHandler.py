from asyncio.windows_events import NULL
import random
import numpy as np
from Data import Data
from os.path import exists


class DataSetHandler:
#region Dataset with single feature
    def GenerateContinuousDataSet_oneProperty(self, filename, minX, maxX, minY, maxY, sampleCount):
        if(exists(filename)):
            raise Exception("File [{name}] is already created".format(name=filename))
        file = open(filename, "w")
        file.write("x\ty\n")

        for i in range(0,sampleCount):
            x=random.gauss((minX+maxX)/2,(minX+maxX)/2)
            y=random.gauss((minY+maxY)/2,(minY+maxY)/2)
            file.write("{x}\t{y}\n".format(x=x, y=y))
        file.close()

    def ReadContinuousDataSet_oneProperty(self, filename):
        dataSet=[]

        if(exists(filename)==False):
            raise Exception("File [{name}] doesn't exist".format(name=filename))
        file = open(filename, "r")

        for line in file.readlines():
            splittedLine=line.split('\t')
            if(splittedLine[0]=="x"):
                continue
            data=Data(float(splittedLine[0]),float(splittedLine[1]))
            dataSet.append(data)

        return dataSet

#endregion

#region Dataset with single feature

    def GenerateContinuousDataSet_multipleProperties(self, filename, minXArray, maxXArray, minY, maxY, sampleCount):
        if(exists(filename)):
            raise Exception("File [{name}] is already created".format(name=filename))
        file = open(filename, "w")
        initialLine=""
        for i in range(len(minXArray)):
            initialLine+="x{i}\t".format(i=i)
        initialLine+="y\n"
        file.write(initialLine)

        for i in range(0,sampleCount):
            x=[]
            for i in range(len(minXArray)):
                x.append(random.gauss((minXArray[i]+maxXArray[i])/2,(minXArray[i]+maxXArray[i])/2))
            y=random.gauss((minY+maxY)/2,(minY+maxY)/2)
            line=""
            for property in x:
                line+=("{property}\t").format(property=property) 
            file.write("{line}{y}\n".format(line=line, y=y))
        file.close()

    def ReadContinuousDataSet_multipleProperties(self, filename):
        dataSet=[]

        if(exists(filename)==False):
            raise Exception("File [{name}] doesn't exist".format(name=filename))
        file = open(filename, "r")

        for line in file.readlines():
            splittedLine=line.split('\t')
            if(splittedLine[0]=="x0"):
                continue
            data=[]
            for n in splittedLine:
                data.append(float(n))
            dataSet.append(data)

        return np.array(dataSet)

    def DataSetToArrayList(self,dataSet):
        newDataSet=[]
        for data in dataSet:
            newData=[]
            newData.append(data.x)
            newData.append(data.y)
            newDataSet.append(newData)  
        return newDataSet

#endregion