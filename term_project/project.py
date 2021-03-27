import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import copy
import math

# blue, orange, green, red, purple
labels = ["Group A", "Group B", "Group C", "Group D", "Group E"]
means = [(-1, -1, -1), (0, 0, 0), (1, 1, 2), (3, 4, -1), (4, 4, 4)]
covs = [
    [[0.1, 0, 0], [0, 0.3, 0], [0, 0, 0.4]],
    [[0.4, 0, 0], [0, 0.2, 0], [0, 0, 0.3]],
    [[0.2, 0, 0], [0, 0.5, 0], [0, 0, 0.6]],
    [[0.6, 0, 0], [0, 0.2, 0], [0, 0, 0.2]],
    [[0.4, 0, 0], [0, 0.1, 0], [0, 0, 0.1]],
]

data = {
    "Group A": np.random.multivariate_normal(means[0], covs[0], 300),
    "Group B": np.random.multivariate_normal(means[1], covs[1], 300),
    "Group C": np.random.multivariate_normal(means[2], covs[2], 300),
    "Group D": np.random.multivariate_normal(means[3], covs[3], 300),
    "Group E": np.random.multivariate_normal(means[4], covs[4], 300),
}
# 그래프 보이기
# blue, orange, green, red, purple
markers = ["o", "x", "1", "+", "s"]

fig = plt.figure()
ax = fig.gca(projection="3d")
for i, label in enumerate(labels):
    X = data[label][:, 0]
    Y = data[label][:, 1]
    Z = data[label][:, 2]
    ax.scatter(X, Y, Z, s=10, marker=markers[i])
# plt.show()

data_all = []
for label in labels:
    for d in data[label]:
        data_all.append(list(d))

k_means_model = KMeans(n_clusters=5)
k_means_model.fit(data_all)
# print("original centers : ", means)
k_means_center = k_means_model.cluster_centers_
for i in range(5):
    for j in range(i + 1, 5):
        if k_means_center[i][0] > k_means_center[j][0]:
            temp = copy.deepcopy(k_means_center[i])
            k_means_center[i] = copy.deepcopy(k_means_center[j])
            k_means_center[j] = temp
k_means_center_with_label = []
for i, k in enumerate(k_means_center):
    k_means_center_with_label.append((i, k))

num = 3
max_dist = [
    [
        num * math.sqrt(covs[0][0][0]),
        num * math.sqrt(covs[0][1][1]),
        num * math.sqrt(covs[0][2][2]),
    ],
    [
        num * math.sqrt(covs[1][0][0]),
        num * math.sqrt(covs[1][1][1]),
        num * math.sqrt(covs[1][2][2]),
    ],
    [
        num * math.sqrt(covs[2][0][0]),
        num * math.sqrt(covs[2][1][1]),
        num * math.sqrt(covs[2][2][2]),
    ],
    [
        num * math.sqrt(covs[3][0][0]),
        num * math.sqrt(covs[3][1][1]),
        num * math.sqrt(covs[3][2][2]),
    ],
    [
        num * math.sqrt(covs[4][0][0]),
        num * math.sqrt(covs[4][1][1]),
        num * math.sqrt(covs[4][2][2]),
    ],
]

test_data = {
    "Group A": np.random.multivariate_normal(means[0], covs[0], 100),
    "Group B": np.random.multivariate_normal(means[1], covs[1], 100),
    "Group C": np.random.multivariate_normal(means[2], covs[2], 100),
    "Group D": np.random.multivariate_normal(means[3], covs[3], 100),
    "Group E": np.random.multivariate_normal(means[4], covs[4], 100),
}

fig1 = plt.figure()
ax1 = fig1.gca(projection="3d")
for i, label in enumerate(labels):
    X = test_data[label][:, 0]
    Y = test_data[label][:, 1]
    Z = test_data[label][:, 2]
    ax1.scatter(X, Y, Z, s=10, marker=markers[i])


# test_result[0]: 올바르게 labeling, test_result[1]: data와의 distance가 다른 mean보다 크다., test_result[2]: 데이터 와의 거리가 max_dist를 넘어간다.
test_result = [[0, 0, 0] for _ in range(5)]
for idx, label in enumerate(labels):
    data_set = test_data[label]
    for d in data_set:
        x, y, z = d
        min_dist_center = (-1, [9999, 9999, 9999])
        for i, center in k_means_center_with_label:
            dist_x = (center[0] - x) * (center[0] - x)
            dist_y = (center[1] - y) * (center[1] - y)
            dist_z = (center[2] - z) * (center[2] - z)
            _, temp = min_dist_center
            temp_dist_x = (temp[0] - x) * (temp[0] - x)
            temp_dist_y = (temp[1] - y) * (temp[1] - y)
            temp_dist_z = (temp[2] - z) * (temp[2] - z)
            if dist_x + dist_y + dist_z < temp_dist_x + temp_dist_y + temp_dist_z:
                min_dist_center = (i, center)
        min_idx, min_center = min_dist_center
        if idx == min_idx:
            if (
                pow(min_center[0] - x, 2) / max_dist[min_idx][0]
                + pow(min_center[1] - y, 2) / max_dist[min_idx][1]
                + pow(min_center[2] - z, 2) / max_dist[min_idx][2]
                <= 1
            ):
                test_result[idx][0] += 1
            else:
                test_result[idx][2] += 1
        else:
            test_result[idx][1] += 1
# blue, orange, green, red, purple
print(
    "[올바르게 labeling., data와의 distance가 다른 mean cluster보다 크다., 최소거리에서의 label은 맞았지만 데이터 와의 거리가 max_dist를 넘어간다.]"
)
print(test_result)


# Randomly Distributed Data
random_data = np.random.multivariate_normal(
    [0, 0, 0], [[100, 0, 0], [0, 100, 0], [0, 0, 100]], 100
)

fig2 = plt.figure()
ax2 = fig.gca(projection="3d")
X_random = random_data[:, 0]
Y_random = random_data[:, 1]
Z_random = random_data[:, 2]
ax2.scatter(X_random, Y_random, Z_random, s=10)

random_test_result = [0, 0, 0, 0, 0]
for d in random_data:
    x, y, z = d
    min_dist_center = (-1, [9999, 9999, 9999])
    for i, center in k_means_center_with_label:
        dist_x = (center[0] - x) * (center[0] - x)
        dist_y = (center[1] - y) * (center[1] - y)
        dist_z = (center[2] - z) * (center[2] - z)
        _, temp = min_dist_center
        temp_dist_x = (temp[0] - x) * (temp[0] - x)
        temp_dist_y = (temp[1] - y) * (temp[1] - y)
        temp_dist_z = (temp[2] - z) * (temp[2] - z)
        if dist_x + dist_y + dist_z < temp_dist_x + temp_dist_y + temp_dist_z:
            min_dist_center = (i, center)
    min_idx, min_center = min_dist_center
    if (
        pow(min_center[0] - x, 2) / max_dist[min_idx][0]
        + pow(min_center[1] - y, 2) / max_dist[min_idx][1]
        + pow(min_center[2] - z, 2) / max_dist[min_idx][2]
        <= 1
    ):
        random_test_result[min_idx] += 1
        print(d, " is in cluster num(" + str(min_idx) + ")")


print(
    "Random distribution N([0, 0, 0], sigma_x^2=100, sigma_y^2=100, sigma_z^2=100) Result : ",
    str(random_test_result),
    "are in clusters[0, 1, 2, 3, 4]",
)

plt.show()
