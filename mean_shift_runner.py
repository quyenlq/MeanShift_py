import mean_shift as ms
import numpy as np
import math
import  pdb
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt
from mean_shift_utils import get_centroids,mse

def plot_solution(c,gt,data):
    plt.figure()
    plt.scatter(c[:, 0], c[:, 1], marker='.', c='r')
    plt.scatter(data[:, 0], data[:, 1], marker='.', c='g')
    plt.scatter(gt[:, 0], gt[:, 1], marker='.', c='m', )
    plt.show()

def ci(cens,gt,k):
    # cens = np.array([c.xy_ for c in sol.centroids_])
    return max(one_way_ci(cens,gt,k),one_way_ci(gt,cens,k))

def one_way_ci(cens_A, cens_B, n_clus):
    sizeA = cens_A.size/2
    sizeB = cens_B.size/2
    biggestSize = np.max([sizeA,sizeB,n_clus])
    orphans = np.ones(biggestSize);
    for cenA in cens_A:
        dist = []
        for cenB in cens_B:
            d = np.dot(cenA-cenB,cenA-cenB)
            dist.append(d);
        mapTo = dist.index(min(dist))
        orphans[mapTo]=0;
    return sum(orphans)


def main(arg):
    if arg=='full':
        print("Use full dataset, might be slow")
        files=files_full
        k=k_full
        kernel = kernel_full
    elif arg=='test':
        print("Testing algorith, only use S1")
        files=files_test
        k=k_test
        kernel = kernel_test
    else:
        print("Lightweight mode, skip Birch1 and Birch2")
        files = files_lightweight
        k = k_lightweight
        kernel = kernel_lightweight
    for f,c,knl in zip(files,k,kernel):
        print 'Running data set %s' %f
        fi = open(f+'.txt')
        fi_gt = open(f+'-gt.txt')
        data = np.loadtxt(fi)
        gt = np.loadtxt(fi_gt)
        reference_points = data
        # pdb.set_trace()
        mean_shifter = ms.MeanShift()
        # kernel_bandwidth = look_distance
        kernel_bandwidth = knl
        print("Kernel bandwidth: %d" %kernel_bandwidth)
        mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth=kernel_bandwidth)
        centroids=get_centroids(mean_shift_result)
        print("Number of clusters: ", max(mean_shift_result.cluster_ids))
        CI = ci(centroids,gt,c)
        MSE = mse(data,centroids, mean_shift_result.cluster_ids)
        print("FINISH DATASET %s CI:=%d, MSE=%.2f" % (f, CI, MSE))



k_full = [15,15,15,15,20,35,50,8,100,100,16];
files_full = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','birch1', 'birch2','dim32']
kernel_full =[50000,50000,38000,38000,2000,1000,500,7000,500,500,20]
k_lightweight = [15,15,15,15,20,35,50,8,16];
files_lightweight = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','dim32']
kernel_lightweight =[50000,50000,38000,38000,2000,1000,500,7000,20]
k_test = [15]
files_test = ['s1']
kernel_test =[50000]
from sys import argv
if len(argv)==2 and (argv[1] == 'full' or argv[1] == 'test'):
    main(argv[1])
else:
    main('lightweight')



