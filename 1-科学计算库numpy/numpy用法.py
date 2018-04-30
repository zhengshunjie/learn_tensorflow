import numpy

world_alcohol = numpy.genfromtxt("world_alcohol.txt", delimiter="," ,dtype=str)
print(type(world_alcohol))

#The numpy.array() function can take a list or list of lists as input. When we input a list, we get a one-dimensional array as a result:
vector = numpy.array([5, 10, 15, 20])
#When we input a list of lists, we get a matrix as a result:
matrix = numpy.array([[5, 10, 15], [20, 25, 30], [35, 40, 45]])
print(vector)
print(matrix)

vector = numpy.array([1,2,3,4])
print(vector.shape)
matrix=numpy.array([[5,10,15],[20,30,40]])
print(matrix.shape)

numbers = numpy.array([1, 2, 3, 4.0])
print(numbers.dtype)