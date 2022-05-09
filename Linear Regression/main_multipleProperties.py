from DataSetHandler import DataSetHandler

from LinearRegression import LinearRegression
from OptimizeGD import OptimizeGD

from DataVisualizer import DataVisualizer

from NormalEquation import NormalEquation

minXArray,maxXArray,minY,maxY=[-17,0],[5,20],7,10
sampleCount=5

dataSetHandler=DataSetHandler()
#dataSetHandler.GenerateContinuousDataSet_multipleProperties("Linear Regression/ContinuousDataSet_multipleProperties 1.txt",minXArray,maxXArray,minY,maxY,sampleCount)
dataSet=dataSetHandler.ReadContinuousDataSet_multipleProperties("Linear Regression/ContinuousDataSet_multipleProperties 1.txt")
bias=1
weightArray=[bias,1,1]

opt=OptimizeGD()
#print(opt.MeanNormalizaation(dataSet))
lr=LinearRegression()
J=lr.CostFunction_multiDemensions(dataSet,weightArray)
#print(J)
jHistory,weightArray=lr.GradientDescent_multiDimensions(dataSet,weightArray)
J=lr.CostFunction_multiDemensions(dataSet,weightArray)
#print(J)
print(weightArray)

ne=NormalEquation()
NE_weights=ne.NormalEquation_multiDimensions(dataSet)
print(NE_weights)

dataVisualizer=DataVisualizer()
dataVisualizer.ShowJHistory(jHistory)