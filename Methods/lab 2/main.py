import numpy
import pandas
import openpyxl


def chebyshev_norm(vector):
    """Returns Chebyshev's norm of vector"""
    abs_vector = [abs(val) for val in vector]
    return max(abs_vector)


def vector_subtract(vec1, vec2):
    """Returns vector that is result of subtracting 2 vectors"""
    vec3 = []
    for i in range(len(vec1)):
        vec3.append(vec1[i] - vec2[i])
    return vec3


A_matrix = numpy.array([        # from A*x = b
    [2.389, 0.273, 0.126, 0.418],
    [0.329, 2.796, 0.179, 0.278],
    [0.186, 0.275, 2.987, 0.316],
    [0.197, 0.219, 0.274, 3.127]
])
b_vector = numpy.array([0.144, 0.297, 0.529, 0.869])    # from A*x = b
B_matrix = [        # from x = Bx + c
     [-1/2.389 * val for val in [0, 0.273, 0.126, 0.418]],
     [-1/2.796 * val for val in [0.329, 0,  0.179, 0.278]],
     [-1/2.987 * val for val in [0.186, 0.275, 0, 0.316]],
     [-1/3.127 * val for val in [0.197, 0.219, 0.274, 0]]]
c_vector = [0.144/2.389, 0.297/2.796, 0.529/2.987, 0.869/3.127]     # from x = Bx + c


def zeydel_method(B, c, norm, epsilon, starting_val=None):
    """Solve system of linear equations x = B*x + c using Zeydel method"""
    if not starting_val:
        # Setting vector c as starting value
        prev = list(c)
    else:
        prev = list(starting_val)
    x = list(prev)
    n = len(x)
    # Variables for displaying information in table
    iteration = 1
    # Array that holds information for displaying in table
    info_arr = [[0] + list(c)]
    while True:

        for i in range(n):
            x_i = 0
            # Finding b_i0 * x_0 + b_i1 * x_1 + ... + b_in * x_n sum and storing in x_i
            for j in range(n):
                x_i += B[i][j] * x[j]

            # Finally, value of the coordinate of vector is x[i] = b_i0 * x_0 + b_i1 * x_1 + ... + b_in * x_n + c_i
            x[i] = x_i + c[i]

        vector_subtract_norm = norm(vector_subtract(x, prev))    # ||x^(k+1) - x^(k)||
        # Getting iteration, x_1, x_2, ..., x_n, ||x^(k+1) - x^(k)|| in a list to use in table
        info_arr.append([iteration] + list(x) + [vector_subtract_norm])

        # Break criteria ||x^(k+1) - x^(k)|| <= eps
        if vector_subtract_norm <= epsilon:

            # Formatting numbers
            for k in range(len(info_arr)):
                info_arr[k] = ["{:.6f}".format(value) if isinstance(value, float) else value for value in info_arr[k]]

            # Making .xlsx table
            df = pandas.DataFrame(info_arr)
            df.to_excel("output.xlsx",
                        header=["Iteration", "$x_1$", "$x_2$", "$x_3$", "$x_4$", "$||x^{(k+1)} - x^{(k)}||$"],
                        index=None)
            return x

        else:
            prev = list(x)
            iteration += 1


x = zeydel_method(B_matrix, c_vector, chebyshev_norm, 0.00001)      # Our solution
numpy_x = numpy.linalg.solve(A_matrix, b_vector)    # Numpy function solution
print("Our solution:", x)
print("Numpy function's solution:", numpy_x)

Ax_vector = numpy.matmul(A_matrix, x)   # A*x
b_minus_Ax_vector = vector_subtract(b_vector, Ax_vector)    # b-A*x
b_minus_Ax_vector = ["{:.6f}".format(coord) for coord in b_minus_Ax_vector]     # Format nicely b-A*x vector
print("b-A*x: ", b_minus_Ax_vector)
# Our solution with starting vector (10,20,30,40)
y = zeydel_method(B_matrix, c_vector, chebyshev_norm, 0.00001, [10, 20, 30, 40])
print("Our solution with (10, 20, 30, 40) starting vector:", y)
