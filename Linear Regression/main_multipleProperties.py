from DataSetHandler import DataSetHandler
from LinearRegression import LinearRegression
from DataVisualizer import DataVisualizer
from OptimizeGD import OptimizeGD

minXArray,maxXArray,minY,maxY=[-17,0],[5,20],7,10
sampleCount=5

dataSetHandler=DataSetHandler()
#dataSetHandler.GenerateContinuousDataSet_multipleProperties("Linear Regression/ContinuousDataSet_multipleProperties 1.txt",minXArray,maxXArray,minY,maxY,sampleCount)
dataSet=dataSetHandler.ReadContinuousDataSet_multipleProperties("Linear Regression/ContinuousDataSet_multipleProperties 1.txt")
bias=1
weightArray=[bias,1,1]

opt=OptimizeGD()
print(opt.MeanNormalizaation(dataSet))
lr=LinearRegression()
J=lr.CostFunction_multiDemension(dataSet,weightArray)
print(J)
jHistory,weightArray=lr.GradientDescent_multiDimension(dataSet,weightArray)
J=lr.CostFunction_multiDemension(dataSet,weightArray)
print(J)

print(weightArray)
dataVisualizer=DataVisualizer()
dataVisualizer.ShowJHistory(jHistory)