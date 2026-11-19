# Site profiles

Clusters: a site -- its platform, fabric family, and the site defaults a run gets when it passes
none (storage roots, subnet). A run names one with MAD_CLUSTER (an entry here) or MAD_CLUSTER_FILE
(a file of the same shape, kept with the site rather than in this public repository); cluster.sh
applies it before its own fallbacks, and anything the run passes wins. These entries are examples.

One file per site; the file name is the name a run gives in `MAD_CLUSTER`. A site keeps its own
profile outside this repository and passes `MAD_CLUSTER_FILE`.
