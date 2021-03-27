import random
import numpy as np

point = []
for i in range(-5, 7):
    x = i
    y = 2 * x - 1 + np.random.normal(0, 2)
    point.append((x, y))


def best_fitting():
    rand_point = []
    while len(rand_point) < 6:
        idx = random.randrange(0, 12)
        if idx not in rand_point:
            rand_point.append(idx)
    matrix = []
    y_vector = []
    for i in rand_point:
        x, y = point[i]
        matrix.append([x, 1])
        y_vector.append([y])
    A = np.matrix(matrix)
    B = np.matrix(y_vector)

    X = np.linalg.inv(A.transpose() * A) * A.transpose() * B
    X = np.asarray(X)
    a = X[0][0]
    b = X[1][0]

    error = 0
    for i in rand_point:
        x, y = point[i]
        y_best = a * x + b
        error += (y_best - y) * (y_best - y)
    return (a, b, error)


best = (999999, 99999, 999999999)
for i in range(10000):
    (a, b, error) = best_fitting()
    if error < best[2]:
        best = (a, b, error)

print("finding (a, b, error) for 6 samples about minimum error : ", best)

# 샘플 모두를 사용한 least_square
mat_all = []
y_all = []
for (x, y) in point:
    mat_all.append([x, 1])
    y_all.append([y])
mat_all = np.matrix(mat_all)
y_all = np.matrix(y_all)

X_all = np.linalg.inv(mat_all.transpose() * mat_all) * mat_all.transpose() * y_all
X_all = np.asarray(X_all)
a_all = X_all[0][0]
b_all = X_all[1][0]

error_all = 0
for (x, y) in point:
    y_all_best = a_all * x + b_all
    error_all += (y_all_best - y) * (y_all_best - y)
print("Least Square Error for all Samples(12) (a,b,c) : ", a_all, b_all, error_all)
