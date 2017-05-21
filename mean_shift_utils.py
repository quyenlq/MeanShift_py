import math
import numpy as np

def euclidean_dist(pointA, pointB):
    if(len(pointA) != len(pointB)):
        raise Exception("expected point dimensionality to match")
    total = float(0)
    for dimension in range(0, len(pointA)):
        total += (pointA[dimension] - pointB[dimension])**2
    return math.sqrt(total)

def gaussian_kernel(distance, bandwidth): 
    euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    val = (1/(bandwidth*math.sqrt(2*math.pi))) * np.exp(-0.5*((euclidean_distance / bandwidth))**2)
    return val


def multivariate_gaussian_kernel(distances, bandwidths):

    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths,2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * math.pi), (dim/2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val



def get_centroids(ms_result):
    pts = ms_result.shifted_points
    dimension = pts[0].size
    lbls = ms_result.cluster_ids
    n_cens = max(lbls)+1
    cens = []
    for i in range(n_cens):
        part = pts[lbls==i]
        cens = np.append(cens, np.mean(part,axis=0).tolist(), axis=0)
    return cens.reshape((n_cens,dimension))

def mse(data, centroids,lbls):
    k = max(lbls)+1
    TSE=0
    for i in range(k):
        c = centroids[i]
        pts = data[lbls==i]
        for p in pts:
            e = np.dot(p-c,p-c)
            TSE+=e
    return TSE/data.size
