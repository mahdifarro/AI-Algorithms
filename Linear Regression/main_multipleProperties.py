from DataSetHandler import DataSetHandler

from LinearRegression import LinearRegression
from OptimizeGD import OptimizeGD

from DataVisualizer import DataVisualizer

from NormalEquation import NormalEquation

minXArray,maxXArray,minY,maxY=[5,0],[10,15],7,10
sampleCount=20

dataSetHandler=DataSetHandler()
# dataSetHandler.GenerateContinuousDataSet_multipleProperties("Linear Regression/ContinuousDataSet_multipleProperties 1.txt",minXArray,maxXArray,minY,maxY,sampleCount)
dataSet=dataSetHandler.ReadContinuousDataSet_multipleProperties("Linear Regression/ContinuousDataSet_multipleProperties 1.txt")
bias=1
weightArray=[bias,1,1]

opt=OptimizeGD()
#print(opt.MeanNormalizaation(dataSet))
lr=LinearRegression()
J=lr.CostFunction_ContinuousMultiDemensions(dataSet,weightArray)
#print(J)
jHistory,weightArray=lr.GradientDescent_ContinuousMultiDimensions(dataSet,weightArray)
J=lr.CostFunction_ContinuousMultiDemensions(dataSet,weightArray)
#print(J)
print(weightArray)
j=lr.CostFunction_ContinuousMultiDemensions(dataSet,weightArray)
print(j)
ne=NormalEquation()
NE_weights=ne.NormalEquation_multiDimensions(dataSet)
print("NE")
print(NE_weights)
j=lr.CostFunction_ContinuousMultiDemensions(dataSet,NE_weights)
print(j)

dataVisualizer=DataVisualizer()
dataVisualizer.ShowJHistory(jHistory)