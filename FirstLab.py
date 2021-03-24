import numpy
from math import exp

a, b = map(float, input("Input A, B: ").strip().split())

A = numpy.array([[101, 250], [40, 101]])
E = numpy.array([[1, 0], [0, 1]])
eigv = numpy.linalg.eig(A)
print(a, b, eigv)

error = 1
delta = 1e-6
# h = 0.00000001
res_grid = [x / 10 for x in range(11)]


def sol(x):
    return (-a / 10 + b / 4) * numpy.asarray([-5, 2]) * exp(x) + (a / 10 + b / 4) * numpy.asarray([5, 2]) * exp(201 * x)


solutions = [sol(x) for x in res_grid]
sol1, sol2 = zip(*solutions)
print("Solutions (precise):",  solutions)
numbers = [range(11)]
y1 = []
y2 = []
err1 = []
err2 = []
h = 0.1
while h > 0.00000001:
    grid = [x * h for x in range(int(1 / h + 1))]
    y = [numpy.asarray([a, b]), ]
    for idx in range(1, len(grid)):
        y.append(numpy.linalg.solve(E - A * h, y[idx - 1]))
    y_fin = [y[int(x / h)] for x in res_grid]
    y1, y2 = zip(*y_fin)
    errors = [abs(solutions[i] - y_fin[i]) for i in range(11)]
    err1, err2 = zip(*errors)
    error = max(max(err1), max(err2)) / max(max(y1), max(y2))
    print(f'h = {h}, error = {error}')
    h /= 2


results = zip(range(11), res_grid, y1, sol1, err1, y2, sol2, err2)
print(*results, sep='\n')
