import numpy as np

class NormalEquation:

    def NormalEquation_multiDimensions(self, dataSet):
        m=len(dataSet)
        dataSet=np.array(dataSet)
        x=dataSet[:,:-1]
        onesArray=np.ones((m,1))
        x = np.append(onesArray,x,axis=1)
        y=dataSet[:,-1]
        pinv=np.linalg.pinv(x.transpose().dot(x))
        weights=(pinv.dot(x.transpose())).dot(y)
        return weights