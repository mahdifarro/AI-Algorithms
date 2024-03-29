import numpy as np

class LinearRegression:

#region Continuous Dataset with single feature

    def CostFunction_Continuous2D(self, dataSet, weightArray):
        m=len(dataSet)
        x=np.array(dataSet)[:,0]
        y=np.array(dataSet)[:,-1]
        predictions=x.dot(weightArray[0])+weightArray[1]
        cost=(1/(2*m))*np.sum(np.square(y-predictions))
        return cost

    def GradientDescent_Continuous2D(self, dataSet, weightArray, alpha=0.0001, iterations=100):
        jHistory=[]
        m=len(dataSet)
        x=np.array(dataSet)[:,0]
        y=np.array(dataSet)[:,-1]
        for it in range(iterations):
            jHistory.append(self.CostFunction_Continuous2D(dataSet, weightArray))
            
            predictions=x.dot(weightArray[0])+weightArray[1]
            weightArray[1]=weightArray[1]-np.sum((alpha/m)*(predictions-y))
            weightArray[0]=weightArray[0]-np.sum((alpha/m)*(x.T.dot(predictions-y)))
        return jHistory

#endregion

#region Continuous Dataset with multiple features

    def CostFunction_ContinuousMultiDemensions(self, dataSet, weightArray):
        m=len(dataSet)
        x=dataSet[:,:-1]
        onesArray=np.ones((m,1))
        x = np.append(onesArray,x,axis=1)
        y=dataSet[:,-1]
        predictions=x.dot(np.array(weightArray))
        cost=(1/(2*m))*np.sum(np.square(y-predictions))
        return cost

    def GradientDescent_ContinuousMultiDimensions(self, dataSet, weightArray, alpha=0.00001, iterations=100000):
        jHistory=[]
        m=len(dataSet)
        x=dataSet[:,:-1]
        onesArray=np.ones((m,1))
        x = np.append(onesArray,x,axis=1)
        y=dataSet[:,-1]
        for it in range(iterations):
            jHistory.append(self.CostFunction_ContinuousMultiDemensions(dataSet, weightArray))
            
            predictions=x.dot(np.array(weightArray))
            weightArray-=np.sum((alpha/m)*(x.T.dot(predictions-y)))
        return jHistory,weightArray

#endregion