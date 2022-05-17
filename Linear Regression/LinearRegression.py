import math
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
        y=dataSet[:,-1]
        onesArray=np.ones((m,1))
        x = np.append(onesArray,x,axis=1)
        predictions=x.dot(np.array(weightArray))
        cost=(1/(2*m))*np.sum(np.square(y-predictions))
        return cost

    def GradientDescent_ContinuousMultiDimensions(self, dataSet, weightArray, alpha=0.001, iterations=100000):
        jHistory=[]
        m=len(dataSet)
        x=dataSet[:,:-1]
        onesArray=np.ones((m,1))
        x = np.append(onesArray,x,axis=1)
        y=dataSet[:,-1]
        i=0
        for it in range(iterations):
            jHistory.append(self.CostFunction_ContinuousMultiDemensions(dataSet, weightArray))
            
            predictions=x.dot(np.array(weightArray))
            weightArray-=((alpha/m)*(x.T.dot(predictions-y)))
        return jHistory,weightArray

#endregion

#region Discrete Dataset with single feature

    def CostFunction_Discrete2D(self, dataSet, weightArray):
        m=len(dataSet)
        x=np.array(dataSet)[:,0]
        y=np.array(dataSet)[:,-1]
        z=x.dot(weightArray[0])+weightArray[1]
        eArray=np.ones((m,1)).dot(math.e)
        eArray=eArray.T
        h=1/(1+np.power((eArray),-z))
        h=h.T
        for i in range(len(y)):
            if h[i]==0 and y[i]==1:
                h[i]=math.pow(10,-8)
            elif h[i]==1 and y[i]==0:
                h[i]=math.pow(10,-8)

        predictions=h.flatten()
        cost=(-1/m)*np.sum(y.dot(np.log(predictions))+(1-y).dot(np.log(1-predictions)))

        return cost

        
    def GradientDescent_Discrete2D(self, dataSet, weightArray, alpha=0.001, iterations=1000):
        jHistory=[]
        m=len(dataSet)
        x=np.array(dataSet)[:,0]
        y=np.array(dataSet)[:,-1]
        for it in range(iterations):
            jHistory.append(self.CostFunction_Continuous2D(dataSet, weightArray))
            
            m=len(dataSet)
            x=np.array(dataSet)[:,0]
            y=np.array(dataSet)[:,-1]
            z=x.dot(weightArray[0])+weightArray[1]
            eArray=np.ones((m,1)).dot(math.e)
            eArray=eArray.T
            h=1/(1+np.power((eArray),-z))
            h=h.T
            for i in range(len(y)):
                if h[i]==0 and y[i]==1:
                    h[i]=math.pow(10,-8)
                elif h[i]==1 and y[i]==0:
                    h[i]=math.pow(10,-8)
            h=h.flatten()

            weightArray[1]=weightArray[1]-np.sum((alpha/m)*(h-y))
            weightArray[0]=weightArray[0]-np.sum((alpha/m)*(x.T.dot(h-y)))
        return jHistory

#endregion
