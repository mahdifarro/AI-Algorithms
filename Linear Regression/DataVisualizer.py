import matplotlib.pyplot as plt
import numpy as np

class DataVisualizer:
    
    def ShowDataSetList_2D(self,dataSet):
        plt.scatter(np.array(dataSet)[:,0], np.array(dataSet)[:,-1])  
        plt.show()

        plt.hist(np.array(dataSet)[:,-1],bins=200)
        plt.show()

    def ShowLinearEquation_2D(self, dataSet, slope, yIntercept):
        x=np.array(dataSet)[:,0]
        y= np.array(dataSet)[:,-1]
        minX,maxX=np.min(x),np.max(x)
        plt.scatter(x,y)  
        x = np.linspace(minX-5,maxX+5,100)
        y = slope*x+yIntercept

        plt.plot(x, y, '-r', label="y={slope}x + {yIntercept}".format(slope=slope, yIntercept=yIntercept))
        plt.title("y={slope}x + {yIntercept}".format(slope=slope, yIntercept=yIntercept))
        plt.xlabel('x', color='#1C2833')
        plt.ylabel('y', color='#1C2833')
        plt.legend(loc='upper left')
        plt.grid()

        plt.show()
        
    def ShowJHistory(self, jHistory):
        iter=np.arange(len(jHistory))
        plt.scatter(iter,jHistory,s=10)
        plt.show()
