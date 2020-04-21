Python implementation of the k-degree anonymity algorithm proposed by Liu and Terzi in:

Liu, Kun, and Evimaria Terzi. "Towards identity anonymization on graphs." In Proceedings of the 2008 ACM SIGMOD international conference on Management of data, pp. 93-106. 2008.

Known issues/bugs:

1) Sampling log(n) edges to swap (as suggested in the paper) yields a significantly lower edge overlap with the original graph

2) The dynamic programming anonymisation algorithm is significantly slower than the greedy one, which shouldn't be the case according to the paper

3) Missing: simultaneous swap
