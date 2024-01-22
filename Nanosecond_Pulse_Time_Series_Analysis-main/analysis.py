# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: my_py310
#     language: python
#     name: my_py310
# ---

# %%
import sys

from os import listdir
from os.path import isfile, join

import multiprocessing

import pandas as pd
import numpy as np

import xlrd

import plotly.express as px
import plotly.graph_objects as go

from scipy import signal
from skimage.restoration import denoise_wavelet

from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler, MaxAbsScaler

from openpyxl import load_workbook
import xlrd

# %%
#import TICC
sys.path.append("../TICC/")
from TICC_solver import TICC 

# %%
# import tkinter
# %matplotlib inline
import matplotlib.pyplot as plt
# %matplotlib inline

# %% [markdown]
# ### Read

# %%
path = "../data_XAUT/Nanosecond_Pulse/20221018-nomex410-0.18mm-10h-50℃-2.6kV/"

f_names = [f for f in listdir(path) if isfile(join(path, f))]

f_nb = 3

df = pd.read_excel(path + f_names[f_nb]) 

# # change variable names
# # voltage1: nanosecond pulse voltage
# # voltage2: local discharge voltage
df = df.rename(columns={"Untitled": "timestamp", 
                        "Untitled 1": "voltage1", 
                        "Untitled 2": "voltage2"})

var_names = ["voltage1", "voltage2"]

# set timestamp to index
df = df.set_index("timestamp")

# %% [markdown]
# ### Nornmalization

# %%
scaler = MaxAbsScaler()

x = df.values 
x_scaled = scaler.fit_transform(x)
df_scaled = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)

# %% [markdown]
# ### Denoise

# %%
# before
plt.plot(df[var_names])

# %%
for name in var_names:
    df[name] = denoise_wavelet(df[name], method='BayesShrink', 
                       mode='soft', wavelet='sym8', 
                       rescale_sigma='True')

# %%
# after
plt.plot(df[var_names])

# %% [markdown]
# ### Visualization

# %%
# original
fig = px.line(df[var_names], markers=False)
fig.show()

# %% [markdown]
# ### TICC

# %%
n_procs = multiprocessing.cpu_count()
w_size = 25

n_cluster = 5

ticc = TICC(window_size=w_size, number_of_clusters=n_cluster, 
            lambda_parameter=5e-2, beta=500, 
            maxIters=100, threshold=5.0e-3, compute_BIC=False,
            write_out_file=False, prefix_string="", 
            num_proc=n_procs)

# %%
(cluster_assignment, cluster_MRFs) = ticc.fit(df_scaled[var_names].to_numpy())


# %% [markdown]
# ### Result Visualization

# %%
#
def generate_start_end(df_index, w_size, cluster_assignment):
    
    clusters = pd.DataFrame(index=df_index)
    clusters["label"] = np.nan
    l = int(w_size/2)
    
    clusters["label"] = np.concatenate((np.ones(w_size-l-1) * cluster_assignment[0], 
                                        cluster_assignment, 
                                        np.ones(l) * cluster_assignment[-1]), axis=None)
    
    clusters["diff"] = clusters["label"].diff()
    clusters["diff"].iloc[0] = 1
    clusters["diff"].iloc[-1] = 1

    start_end = clusters.index[clusters["diff"] != 0.0].tolist()
    
    return clusters, start_end  


# %%
clusters, start_end = generate_start_end(df.index, w_size, cluster_assignment)

# %%
fig1 = go.Figure()

for name in var_names:
    fig1.add_trace(go.Scatter(x=df.index, y=df[name], name=name))


for i in range(len(start_end)-1):
    l = clusters.loc[start_end[i]]["label"]
    
    fig1.add_vrect(x0=start_end[i], x1=start_end[i+1], 
                   fillcolor=px.colors.qualitative.Dark24[int(l)],
                   annotation_text = int(l),
                   opacity=0.3, layer="below", line_width=0)
    
fig1.show()

# %%

# %%

# %%

# %%

# %%
