from DataSetHandler import DataSetHandler
from LinearRegression import LinearRegression
from DataVisualizer import DataVisualizer

minX,maxX,minY,maxY=70,100,7,10
sampleCount=1000

dataSetHandler=DataSetHandler()
#dataSetHandler.GenerateContinuousDataSet_oneProperty("ContinuousDataSet_oneProperty 1.txt",minX,maxX,minY,maxY,sampleCount)
dataSet_2d=dataSetHandler.ReadContinuousDataSet_oneProperty("ContinuousDataSet_oneProperty 1.txt")
dataSet_2d=dataSetHandler.DataSetToArrayList(dataSet_2d)

slope=1
yIntercept=0
weightArray=[slope,yIntercept]

dataVisualizer=DataVisualizer()
dataVisualizer.ShowDataSetList_2D(dataSet_2d)
dataVisualizer.ShowLinearEquation_2D(dataSet_2d,weightArray[0],weightArray[1])

lr=LinearRegression()
J=lr.CostFunction_2D(dataSet_2d,weightArray)
print(J)
jHistory=lr.GradientDescent_2D(dataSet_2d,weightArray)
J=lr.CostFunction_2D(dataSet_2d,weightArray)
print(J)

dataVisualizer.ShowLinearEquation_2D(dataSet_2d,weightArray[0],weightArray[1])
dataVisualizer.ShowJHistory(jHistory)