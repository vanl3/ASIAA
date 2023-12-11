#!/usr/bin/env python
# coding: utf-8

# In[7]:


import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import time
import math
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatterMathtext
from matplotlib.ticker import StrMethodFormatter
import matplotlib.lines as lines
from pylab import fromfile
from multiprocessing import Pool


# In[1]:


# pwd


# In[27]:


path  = '/tiara/ara/data/vle/fargo3d-2.01/fargo3d-2.01/outputs/asiaa_torque1/'
paths = '/tiara/ara/data/vle/movie'



# In[80]:


def make_plot(param): 
    fpath, path, nf = param
            
    phi_dat = np.loadtxt(f'{fpath}domain_x.dat')
    rad_dat  = np.loadtxt(f'{fpath}domain_y.dat')[3:-3]
    
    phi = 2.*np.pi*np.linspace(0,1,len(phi_dat)-1) - np.pi/2.
    rad   = 0.5*(rad_dat[:-1] + rad_dat[1:])
    
    nx = len(phi)
    ny = len(rad)
    
    P, R = np.meshgrid(phi, rad)
    X = R*np.cos(P)
    Y = R*np.sin(P)
    
    rhog_i = fromfile(fpath+'gasdens0.dat').reshape(ny,nx)    
    rhod_i = fromfile(fpath+'dust1dens0.dat').reshape(ny,nx)
    
    rhog   = fromfile(f'{fpath}gasdens{nf}.dat').reshape(ny,nx) 
    rhod1  = fromfile(f'{fpath}dust1dens{nf}.dat').reshape(ny,nx)
    
    gas = rhog/rhog_i
    dust1 = rhod1/rhod_init

    fig, (ax, ax1) = plt.subplots(nrows=1, ncols=2,figsize=(16,5), sharey=True)
    
    ax.pcolormesh(phi_dat, rad_dat, rhog/rhog_i)
    ax.set_ylabel('rad', fontsize=18)
    ax.set_xlabel('phi', fontsize=18)
    ax.set_ylim(0.2,3)
    
    ax1.pcolormesh(phi_dat, rad_dat, rhod1/rhod_i)
    # ax1.set_ylabel('rad', fontsize=18)
    ax1.set_xlabel('phi', fontsize=18)
    ax1.set_ylim(0.2,3)
    
    plt.savefig(f'{path}gddens_{nf}.png', transparent=True, dpi=300, bbox_inches='tight') # dots per inch
    plt.close()


# In[ ]:

# input for parallel jobs
num_cores = 12
frame_range = range(0, 1001)
params = [(path0_old, paths0t, nf) for nf in frame_range]

# Iterate through frames
if len(frame_range) == 1:
    make_plot(params[0])
else:
    if num_cores > 1:
        p = Pool(num_cores) # default number of processes is multiprocessing.cpu_count()
        p.map(make_plot, params)
        p.terminate()
    else:
        for param in params:
            make_plot(param)

