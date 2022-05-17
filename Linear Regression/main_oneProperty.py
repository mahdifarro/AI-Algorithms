from DataSetHandler import DataSetHandler
from LinearRegression import LinearRegression
from DataVisualizer import DataVisualizer

minX,maxX,minY,maxY=70,100,10,60
sampleCount=40

#region Continuous DataSet

# dataSetHandler=DataSetHandler()
# #dataSetHandler.GenerateContinuousDataSet_oneProperty("ContinuousDataSet_oneProperty 1.txt",minX,maxX,minY,maxY,sampleCount)
# dataSet_2d=dataSetHandler.ReadContinuousDataSet_oneProperty("ContinuousDataSet_oneProperty 1.txt")
# dataSet_2d=dataSetHandler.DataSetToArrayList(dataSet_2d)

# slope=1
# yIntercept=0
# weightArray=[slope,yIntercept]

# dataVisualizer=DataVisualizer()
# dataVisualizer.ShowDataSetList_2D(dataSet_2d)
# dataVisualizer.ShowLinearEquation_2D(dataSet_2d,weightArray[0],weightArray[1])

# lr=LinearRegression()
# J=lr.GradientDescent_Continuous2D(dataSet_2d,weightArray)
# print(J)
# jHistory=lr.GradientDescent_Continuous2D(dataSet_2d,weightArray)
# J=lr.GradientDescent_Continuous2D(dataSet_2d,weightArray)
# print(J)

# dataVisualizer.ShowLinearEquation_2D(dataSet_2d,weightArray[0],weightArray[1])
# dataVisualizer.ShowJHistory(jHistory)

#endregion

#region Discrete DataSet

dataSetHandler=DataSetHandler()
# dataSetHandler.GenerateDiscreteDataSet_oneProperty("Linear Regression/DiscreteDataSet_oneProperty 1.txt",minX,maxX,minY,maxY,sampleCount)
dataSet_2d=dataSetHandler.ReadDiscreteDataSet_oneProperty("Linear Regression/DiscreteDataSet_oneProperty 1.txt")
dataSet_2d=dataSetHandler.DataSetToArrayList(dataSet_2d)

slope=1
yIntercept=0
weightArray=[slope,yIntercept]

dataVisualizer=DataVisualizer()
dataVisualizer.ShowLinearEquation_Discrete2D(dataSet_2d,weightArray[0],weightArray[1])

lr=LinearRegression()
J=lr.CostFunction_Discrete2D(dataSet_2d,weightArray)
print(J)
jHistory=lr.GradientDescent_Discrete2D(dataSet_2d,weightArray)
J=lr.CostFunction_Discrete2D(dataSet_2d,weightArray)
print(J)

dataVisualizer.ShowLinearEquation_Discrete2D(dataSet_2d,weightArray[0],weightArray[1])
dataVisualizer.ShowJHistory(jHistory)


#endregion