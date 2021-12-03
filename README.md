# Network_ExtremeValue_Bootstrap

This is the repo for the Bootstrapping Network Max Degree Distribution project which is advised by [Prof. Keith Levin](https://stat.wisc.edu/staff/levin-keith/) at UW Madison

1. Problem Setup
Given a network $G$ with $n$ nodes under random dot product graph model(RDPG) following an unknown latent distribution $F$ on $R^d$, we denote the adjacency matrix as $A_{n \times n}$ and latent positions $X_{n \times d}$ where each row of $X$ is drawn i.i.d. from $F$. We write $(A,X) \sim RDPG(F,n)$. Our goal is to estimate the max degree distribution $EV_n$ of $RDPG(F,n)$ based on the observed network $G$. 
