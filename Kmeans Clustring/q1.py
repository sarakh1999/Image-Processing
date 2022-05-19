import matplotlib.pylab as plt
import random
import numpy as np
from sklearn.cluster import KMeans


def read_data(file_name):
    file = open(file_name, 'r')
    file_list = file.readlines()
    number_of_points = int(file_list[0])
    points = []
    x_s = []
    y_s = []
    for i in range(1, number_of_points + 1):
        x, y = file_list[i].split()
        x1 = float(x)
        y1 = float(y)
        points.append([x1, y1])
        x_s.append(x1)
        y_s.append(y1)
    points = np.array(points)
    fig, ax = plt.subplots()
    ax.scatter(x_s, y_s, color="black")
    plt.savefig('res01.jpg')
    test_with_python_kmeans_module(points)
    return points


def test_with_python_kmeans_module(points):
    kmeans = KMeans(n_clusters=2, random_state=8).fit(points)
    labels = (kmeans.labels_)
    plt.scatter(points[:, 0], points[:, 1], c=labels)
    plt.savefig('test01.jpg')


def select_random_seeds(points):
    centroids = []
    number1 = random.randrange(0, points.shape[0])
    number2 = random.randrange(0, points.shape[0])
    while number2 == number1:
        number2 = random.randrange(0, points.shape[0])
    centroid1, centroid2 = points[number1], points[number2]
    centroids.append(centroid1)
    centroids.append(centroid2)
    centroids = np.array(centroids)
    return centroids


def reassignment(points, centroids):
    cluster_of_each_point = []
    for datum in points:
        cluster_of_each_point.append(np.argmin(np.sum((datum.reshape(1, points.shape[1]) - centroids) ** 2, axis=1)))
    return cluster_of_each_point


def update_centroids(points, cluster_of_each_point, number_of_clusters):
    centroids = []
    for cluster in range(number_of_clusters):
        centroids.append(
            np.mean([points[d] for d in range(points.shape[0]) if cluster_of_each_point[d] == cluster], axis=0))
    return centroids


def check_convergence(points, previous_centroids, new_centroids):
    previous_centroids = np.array(previous_centroids)
    previous_centroids.reshape(2, points.shape[1])
    new_centroids = np.array(new_centroids)
    new_centroids.reshape(2, points.shape[1])
    difference = new_centroids - previous_centroids
    flag = True
    threshold = 0.0000001
    for i in range(2):
        for j in range(points.shape[1]):
            if abs(difference[i][j]) >= threshold:
                flag = False
    return flag


def show_results(points, centroids, name, cluster_of_each_point):
    centroids = np.array(centroids)
    centroids.reshape(2, 2)
    # print("cluster_of_each_point", cluster_of_each_point)
    # print("centroids", centroids)
    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], c=cluster_of_each_point)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', label='centroids')
    plt.savefig(name)


def cast_to_polar_coordinaton(points):
    new_points = []
    for i in range(points.shape[0]):
        x = points[i][0]
        y = points[i][1]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan(y / x)
        new_points.append([r, theta])
    new_points = np.array(new_points)
    return new_points


def kmeans_clustring(new_points, points, name):
    number_of_clusters = 2
    centroids = select_random_seeds(new_points)
    step = 0
    while True:
        step += 1
        cluster_of_each_point = reassignment(new_points, centroids)
        new_centroids = update_centroids(new_points, cluster_of_each_point, number_of_clusters)
        if check_convergence(new_points, centroids, new_centroids):
            break
        else:
            centroids = new_centroids
    show_results(points, centroids, name, cluster_of_each_point)


points = read_data('Points.txt')
kmeans_clustring(points, points, 'res02.jpg')
kmeans_clustring(points, points, 'res03.jpg')
new_points = cast_to_polar_coordinaton(points)
kmeans_clustring(new_points, points, 'res04.jpg')
