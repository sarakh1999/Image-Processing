import numpy as np
import cv2
import time


def create_feature_vector(img):
    length = img.shape[0]
    width = img.shape[1]
    number_of_features = 5
    feature_vector = np.zeros((length, width, number_of_features), dtype=np.uint32)
    normalization_coeff = 0.25
    feature_vector[:, :, :3] = img[:, :, :]
    feature_vector[:, :, 3] = np.arange(length).reshape((length, 1)) * normalization_coeff
    feature_vector[:, :, 4] = np.arange(width).reshape((1, width)) * normalization_coeff
    feature_vector = feature_vector.reshape((-1, number_of_features))
    return feature_vector


def smooth(img):
    kernel_length = 3
    one_matrix = np.full((kernel_length, kernel_length), 1)
    kernel_size = kernel_length * kernel_length
    blur_filter = (1 / kernel_size) * one_matrix
    blured_img = cv2.filter2D(src=img, ddepth=-1, kernel=blur_filter)
    blured_img = blured_img[1:-1, 1:-1]
    return blured_img


def resize_img(img, scale):
    new_y = int(img.shape[1] * scale)
    new_x = int(img.shape[0] * scale)
    scaled_img = cv2.resize(img, (new_y, new_x), interpolation=cv2.INTER_AREA)
    return scaled_img


def initialize_centriods(feature_vector):
    initial_centroids = np.array(feature_vector)
    return initial_centroids


def check_convergence(initial_centroids, previous_centroids):
    delta = initial_centroids - previous_centroids
    differences = np.sum(abs(delta))
    if differences <= 1:
        return True
    return False


def mean_shift(resized_img, feature_vector):
    threshold = 1600
    number_of_iterations = 5
    initial_centroids = initialize_centriods(feature_vector)
    for iteration in range(number_of_iterations):
        new_centroids = []
        number_of_centroids = feature_vector.shape[0]
        for i in range(number_of_centroids):
            delta = feature_vector - initial_centroids[i]
            differences = np.sum(delta ** 2, axis=1)
            neighbours = feature_vector[differences < threshold]
            if len(neighbours) == 0:
                neighbours = [initial_centroids[i]]
            cluster_average = np.average(neighbours, axis=0)
            new_centroids.append(cluster_average)
        new_centroids = np.array(new_centroids)
        previous_centroids = initial_centroids
        initial_centroids = new_centroids
        if check_convergence(initial_centroids, previous_centroids):
            break
    new_img = np.array(initial_centroids[:, :3], dtype=np.uint8).reshape(resized_img.shape)
    return new_img


begin = time.time()
img = cv2.imread('park.jpg')
img = smooth(img)
cv2.imwrite('blur.jpg', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
resized_img = resize_img(img, scale=0.1)
feature_vector = create_feature_vector(resized_img)
segmented_img = mean_shift(resized_img, feature_vector)
# img = smooth(img)
result = resize_img(segmented_img, scale=1)
result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
cv2.imwrite('res05.jpg', result)
end = time.time()
print(end - begin)
