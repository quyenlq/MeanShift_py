import mean_shift as ms
import numpy as np
import math
import  pdb
from sklearn.cluster import estimate_bandwidth
import matplotlib.pyplot as plt

def plot_solution(X,gt):
    plt.figure()
    # cx = np.array([c.xy_ for c in sol.centroids_])
    # for c in sol.centroids_:
    #   partition = sol.get_partition(c.label_)
    #   pxy = np.array([[p.getX(), p.getY()] for p in partition])
    #   # pdb.set_trace()
    #   plt.scatter(pxy[:, 0], pxy[:, 1], s=2, marker='.', c='b')
    plt.scatter(X[:, 0], X[:, 1], marker='.', c='r')
    plt.scatter(gt[:, 0], gt[:, 1], marker='.', c='m', )
    plt.show()

def ci(sol,gt,k):
    cens = np.array([c.xy_ for c in sol.centroids_])
    return max(one_way_ci(cens,gt,k),one_way_ci(gt,cens,k))

def one_way_ci(cens_A, cens_B, n_clus):
    orphans = np.ones(n_clus);
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
        look_distance = look_distance_full
    elif arg=='test':
        print("Testing algorith, only use S1")
        files=files_test
        k=k_test
        look_distance = look_distance_test
    else:
        print("Lightweight mode, skip Birch1 and Birch2")
        files = files_lightweight
        k = k_lightweight
        look_distance = look_distance_lightweight
    for f,c,d in zip(files,k,look_distance):
        print 'Running data set %s' %f
        fi = open(f+'.txt')
        fi_gt = open(f+'-gt.txt')
        data = np.loadtxt(fi)
        gt = np.loadtxt(fi_gt)
        reference_points = data
        mean_shifter = ms.MeanShift()
        kernel_bandwidth = look_distance
        print("Kernel bandwidth: %d" %kernel_bandwidth)
        mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth=kernel_bandwidth)
        mean_shift_centroids = np.unique(mean_shift_result.shifted_points)
        print("Cluster ids: ", np.unique(mean_shift_result.cluster_ids))
        # print("Cluster ids: ", np.unique(mean_shift_result.cluster_ids))
        
        # print "Original Point     Shifted Point  Cluster ID"
        # print "============================================"
        # for i in range(len(mean_shift_result.shifted_points)):
        #     original_point = mean_shift_result.original_points[i]
        #     converged_point = mean_shift_result.shifted_points[i]
        #     cluster_assignment = mean_shift_result.cluster_ids[i]
        #     print "(%5.2f,%5.2f)  ->  (%5.2f,%5.2f)  cluster %i" % (original_point[0], original_point[1], converged_point[0], converged_point[1], cluster_assignment)

        # X=ms(data, d,gt)
        # plot_solution(X,gt)
        # CI = ci(sol,gt,c)
        # print("FINISH DATASET %s CI:=%d, MSE=%.2f" % (f, CI, sol.MSE_))
        # plot_solution(sol,gt)





k_full = [15,15,15,15,20,35,50,8,100,100,16];
files_full = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','birch1', 'birch2','dim32']
look_distance_full =[50000,50000,50000,50000,5000,5000,5000,20000,5000,5000,20]
k_lightweight = [15,15,15,15,20,35,50,8,16];
files_lightweight = ['s1','s2','s3','s4', 'a1', 'a2', 'a3','unbalance','dim32']
look_distance_lightweight =[50000,50000,50000,50000,5000,5000,5000,20000,20]
k_test = [15]
files_test = ['s1']
look_distance_test =[50000]
# np.random.seed(2991)
from sys import argv
if len(argv)==2 and (argv[1] == 'full' or argv[1] == 'test'):
    main(argv[1])
else:
    main('lightweight')


    # reference_points = load_points("data.csv")
    # mean_shifter = ms.MeanShift()
    # mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth = 3)
    
    # print "Original Point     Shifted Point  Cluster ID"
    # print "============================================"
    # for i in range(len(mean_shift_result.shifted_points)):
    #     original_point = mean_shift_result.original_points[i]
    #     converged_point = mean_shift_result.shifted_points[i]
    #     cluster_assignment = mean_shift_result.cluster_ids[i]
    #     print "(%5.2f,%5.2f)  ->  (%5.2f,%5.2f)  cluster %i" % (original_point[0], original_point[1], converged_point[0], converged_point[1], cluster_assignment)

# if __name__ == '__main__':
#     run()