# methods for clustering images
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, MiniBatchKMeans

IMAGES = "Images"
CLUSTERS = "Clusters"
MINIBATCHKMEANS = "MiniBatchKMeans"
KMEANS = "KMeans"
SPECTRALCLUSTERING = "SpectralClustering"
XLABEL = "Clusters number"
YLABEL = "Score"
MAX_ITER = 30000
N_JOBS = 4
N_INIT = 100
FOLDER = "Clustering"

KMEANS_N = 26
MINIBATCHKMEANS_N = 28
SPECTRALCLUSTERING_N = 24



def showKMeans(X, N):
    scores = []
    for number in xrange(N / 6, N / 2):
        clustering = KMeans(n_clusters=number, max_iter=MAX_ITER, n_init=N_INIT, n_jobs=N_JOBS )
        clustering.fit_predict(X)
        scores.append(clustering.score(X))
    plt.plot(scores)
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.show()
    
def writeKmeans(X, number, objectsNames):
    clustering = KMeans(n_clusters=number, max_iter=MAX_ITER, n_init=N_INIT, n_jobs=N_JOBS)
    results = np.array(clustering.fit_predict(X))

    resuldDF = pd.DataFrame({IMAGES:objectsNames, CLUSTERS:results})
    resuldDF.to_csv(FOLDER + "/" + KMEANS+"_"+str(number)+".csv", index=False)

def writeSpectralClustering(X, number, objectsNames):
    clustering = SpectralClustering(n_clusters=number, affinity='nearest_neighbors')
    results = np.array(clustering.fit_predict(X))

    resuldDF = pd.DataFrame({IMAGES:objectsNames, CLUSTERS:results})
    resuldDF.to_csv(FOLDER + "/" + SPECTRALCLUSTERING+"_"+str(number)+".csv", index=False)

def showMiniBatchKMeans(X, N):
    scores = []
    for number in xrange(N / 6, N / 2):
        clustering = MiniBatchKMeans(n_clusters=number, max_iter=MAX_ITER )
        clustering.fit_predict(X)
        scores.append(clustering.score(X))
    plt.plot(scores)
    plt.xlabel(XLABEL)
    plt.ylabel(YLABEL)
    plt.show()
    
def writeMiniBatchKMeans(X, number, objectsNames):
    clustering = MiniBatchKMeans(n_clusters=number, max_iter=MAX_ITER)
    results = np.array(clustering.fit_predict(X))
    
    resuldDF = pd.DataFrame({IMAGES:objectsNames, CLUSTERS:results})
    resuldDF.to_csv(FOLDER + "/" + MINIBATCHKMEANS+"_"+str(number)+".csv", index=False)