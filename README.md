# Influentials
 
 This project explores the idea of *influentials* - a minority of individuals who are capable of influencing an exceptional number of their peers. 
 Specifically, we explore whether *influentials* play a significant role in the spread of innovations/ideas compared with the average indivudal or whether there is another factor at play such as the interrelationship within a community.
 
 ## Analysis
 
 This project contains the following:
 
 - Simulations of the spread of innovations/ideas in a poisson random graph and scale free network under the threshold model.
 
 - Simulations of the spread of innovations/ideas using Facebook/Twitter data.
 
 
## Environment

Install Miniconda/Anaconda and run the following commmand in terminal within the main directory

    conda env create -f binder/environment.yml
    conda activate dask
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    conda install -c plotly -c defaults -c conda-forge "jupyterlab>=1.0" jupyterlab-dash=0.1.0a3
    jupyter labextension install jupyterlab-dash jupyterlab-plotly
    
    
## Sources

### Paper Reference
  Influentials, Networks, and Public Opinion Formation Paper: https://www.uvm.edu/pdodds/research/papers/others/2007/watts2007a.pdf

### Data
  Twitter data: https://snap.stanford.edu/data/ego-Twitter.html
  
  Facebook data: https://snap.stanford.edu/data/ego-Facebook.html
