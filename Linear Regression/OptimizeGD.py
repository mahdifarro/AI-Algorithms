import numpy as np

class OptimizeGD:

    def FeatureSize(self, datas):
        min, max = np.min(datas), np.max(datas)
        size=0
        if(abs(min)>=abs(max)):
            size=abs(min)
        else:
            size=max
        return int(size)

    def FeatureScaling(self, dataset):
        dataSet=np.array(dataSet)
        for i in range(np.array(dataSet).shape[1]-1):
            column=dataSet[:,i]
            dataSet[:,i]=column.dot(1/self.FeatureSize(column))
        return dataSet
        
    def MeanNormalizaation(self, dataSet):
        dataSet=np.array(dataSet)
        for i in range(np.array(dataSet).shape[1]-1):
            column=dataSet[:,i]
            dataSet[:,i]=(column-column.mean()).dot(1/self.FeatureSize(column))
        return dataSet