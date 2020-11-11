#!/usr/bin/env python
# coding: utf-8

# Cluster data using various methods.
#
# Heavily based on https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
# See that page for nice explanations too.
#
# Messed with a bit by Marcus Frean
# Expects to read a datafile consisting of a matrix in which each row is a training item.

import sys, math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import sklearn.cluster as cluster
import time
import pandas as pd

sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 50, 'linewidths':1}


# def plot_clusters(outfile, data, algorithm, args, kwds):
def plot_clusters(outfile, data, algorithm, args, kwds):
    fig = plt.figure()
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data['x'], data['y'], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=14)
    plt.show()
    #plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)
    plt.savefig(outfile)
    print ( '\n  saved image ',outfile )
    plt.close(fig)
    return    


if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Three groups of system parameters.
    # parser.add_argument('-f','--infile', required=True, action="store", help='input data as a .csv file')
    # args = parser.parse_args()
    # out_stem = args.infile.split('.')[0]
    #
    # # Read in some data from a csv file
    # args.infile = 'demo.csv'
    # data = np.genfromtxt(args.infile, float) #, unpack=True)
    # NUM_DATA_ITEMS,NUM_DATA_DIM = data.shape
    data = pd.read_csv("/Users/vini/Desktop/comp471A2/demo_data.csv",dtype=float)
    print(data)
    plt.scatter(data[0], data[1], c='b', **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    # plt.savefig(out_stem+'_RAW.png')
    plt.show()
    
    
    ### See this with great pithy explanations at 
    ### https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html
    plot_clusters('_K2means.png',data, cluster.KMeans, (), {'n_clusters':2})
    plot_clusters('_K4means.png',data, cluster.KMeans, (), {'n_clusters':4})
    plot_clusters('_K6means.png',data, cluster.KMeans, (), {'n_clusters':6})
    plot_clusters('_K8means.png',data, cluster.KMeans, (), {'n_clusters':8})

    # plot_clusters(out_stem+'_MeanShift.png', data, cluster.MeanShift, (0.175,), {'cluster_all':False})
    #
    # plot_clusters(out_stem+'_SpectralClustering.png',data, cluster.SpectralClustering, (), {'n_clusters':6})
    #
    # plot_clusters(out_stem+'_AffinityProp.png',data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})
    #
    # plot_clusters(out_stem+'_AgglomerativeClustering.png',
    #               data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})
    #

    # NOTE. You'll need to install HDBSCAN (see https://hdbscan.readthedocs.io/en/latest/) to see it, as it's not part of sklearn yet. Works frighteningly well though, from the demo.
#    import hdbscan
#    plot_clusters(out_stem+'_HDBSCAN05.png',data, hdbscan.HDBSCAN, (), {'min_cluster_size':5}) 
#    plot_clusters(out_stem+'_HDBSCAN15.png',data, hdbscan.HDBSCAN, (), {'min_cluster_size':15}) 
#    plot_clusters(out_stem+'_HDBSCAN25.png',data, hdbscan.HDBSCAN, (), {'min_cluster_size':25}) 
    

