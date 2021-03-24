import numpy
A = numpy.array([[101, 250], [40, 101]])
E = numpy.array([[1, 0], [0, 1]])
h = 0.1
y0 = numpy.asarray([1, 1])

print(E - A * h)
print(numpy.linalg.solve(E - A * h, y0))