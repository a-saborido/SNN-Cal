# Neuromorphic Readout for Hadronic Calorimeters


## How to Run 

Here is a quick overview of the necessary steps to take in order to run the code contained in this repository. This is not meant to be an exhaustive overview, merely a simple list of instructions for the less tech-savvy audience.

### Get this Repository on your System
First, get a local copy of this repository so that you can run and modify it as you wish. In order to do that, I suggest first [forking](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) it so that you have your own personal version of it, and then [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) it locally on your computer.

> **WARNING!**: The main branch is currently not updated! Please use the *lin_model* branch to enjoy all the functionalities.


### Set up the Python Environment

Second, make sure you have all the packages necessary to run the code. I suggest using a package and environment manager like Anaconda. 

#### Install Miniconda
Install Miniconda on your system, following the information detailed on [this page](https://docs.anaconda.com/miniconda/install/). You can do it easily using these commands on a Linux machine:

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init --all
```

#### Create the Conda Env
Generate a conda environment ([here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) is a list of commands you can use) and download the necessary packages, in particular *numpy*, *PyTorch*, *snnTorch* and *Jupyter Notebook*. <br> If you want the exact environment that I used, run the following command:
```bash
conda env create -f env/environmental_droplet.yml
```
to create the *snn_hgcal* environment with all the needed components.
