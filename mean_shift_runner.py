import mean_shift as ms
import numpy as np
import pdb
# import matplotlib.pyplot as plt
from mean_shift_utils import get_centroids,mse


data_profile = {
	's1': [50000,15],
	's2': [50000,15],
	's3': [38000,15],
	's4': [38000, 15],
	'a1': [2000,20],
	'a2': [1000,35],
	'a3': [500,50],
	'unbalance': [7000,8],
	'birch1': [500,100],
	'birch2': [500,100],
	'unbalance': [25,16]
}


def ci(cens,gt,k):
	return max(one_way_ci(cens,gt,k),one_way_ci(gt,cens,k))

def one_way_ci(cens_A, cens_B, n_clus):
	d = cens_A[0].size
	sizeA = cens_A.size/d
	sizeB = cens_B.size/d
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

def run_mean_shift(f,knl, c):
	print 'Running data set %s' % f
	fi = open(f + '.txt')
	fi_gt = open(f + '-gt.txt')
	data = np.loadtxt(fi)
	gt = np.loadtxt(fi_gt)
	reference_points = data
	mean_shifter = ms.MeanShift()
	kernel_bandwidth = knl
	print("Kernel bandwidth: %d" % kernel_bandwidth)
	mean_shift_result = mean_shifter.cluster(reference_points, kernel_bandwidth=kernel_bandwidth)
	centroids = get_centroids(mean_shift_result)
	print("Number of clusters: ", max(mean_shift_result.cluster_ids)+1)
	CI = ci(centroids, gt, c)
	MSE = mse(data, centroids, mean_shift_result.cluster_ids)
	print("FINISH DATASET %s CI:=%d, MSE=%.2f" % (f, CI, MSE))


def main(arg):
	ds_name = arg[1]
	if ds_name == 'all':
		print("Use full dataset, might be slow")
		for f, p in data_profile.items():
			knl = p[0]  # kernel bandwidth
			k = p[1]  # cluster size
			run_mean_shift(f, knl, k)
	elif ds_name == 'test':
		print("Testing algorith, only use S1")
		f = 's1'
		p = data_profile[f]
		knl = p[0]
		k = p[1]
		run_mean_shift(f, knl, k)
	if ds_name not in data_profile.keys():
		print("Wrong dataset, try dataset name without extension (for example s1, s2 instead of s1.txt")
	else:
		f = ds_name
		p = data_profile[f]
		knl = p[0]
		k = p[1]
		run_mean_shift(f, knl, k)

from sys import argv
if len(argv)==2:
	main(argv)
else:
	print 'ERROR: Wrong arguments: $ %s <dataset_name>"' % (argv[0])