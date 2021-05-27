# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:09:00 2021

@author: jesel
"""
import numpy as np
import tensorflow as tf

import time
import arrow

from point.metrics import Metrics
from point.point_process import PointsData

import gpflow
from gpflow.config import default_float


directory = "D:\GitHub\point\data\data_rff"
expert_seq = np.load(directory + "\data_synth_points.npy")
expert_space = np.load(directory + "\data_synth_space.npy", allow_pickle=True)
expert_sizes = np.load(directory + "\data_synth_sizes.npy", allow_pickle=True)
expert_data = PointsData(expert_sizes, expert_seq, expert_space)

rng = np.random.RandomState(10)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
directory = "D:\GitHub\point\exprmt"
vbb = np.load(directory + "\data_vbb.npy", allow_pickle=True)

print("")
print("VBB")

t0 = time.time()
training_time = sum(vbb[:,1])
params = tf.stack(vbb[:,0])
variance_lst = params[:,-1]
lenghtscale_lst = params[:,-2]

variance = tf.math.reduce_mean(variance_lst[-10])
lenght_scale = tf.math.reduce_mean(lenghtscale_lst[-2])

print("[{}] variance:= : {}".format(arrow.now(), variance.numpy() ))
print("[{}] lenght_scale:= {}".format(arrow.now(), lenght_scale.numpy() ))
print("[{}] training.time:= {}".format(arrow.now(), training_time ))

loss_reward = Metrics(batch_size = 1000, random_state = rng).negative_rewards(expert_data, variance = variance, length_scale = length_scale )
loss_logl = Metrics(batch_size = 1000, random_state = rng).log_likelihood(expert_data, variance = variance, length_scale = length_scale )

print("VBB.Loss")
print("[{}] negative.reward.term : {}".format(arrow.now(), loss_reward ))
print("[{}] negative.log.likelihood : {}".format(arrow.now(), loss_logl))
print("[{}] time : {}".format(arrow.now(), time.time() - t0))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
directory = "D:\GitHub\point\exprmt"
irl = np.load(directory + "\data_irl_b500.npy", allow_pickle=True)

print("")
print("IRL")

t0 = time.time()
params = tf.stack(irl[:,0])
variance_lst = params[:,-1]
lenghtscale_lst = params[:,-2]
training_time = sum(irl[:,1])

variance = tf.math.reduce_mean(variance_lst[-10])
lenght_scale = tf.math.reduce_mean(lenghtscale_lst[-2])

print("[{}] variance:= : {}".format(arrow.now(), variance.numpy() ))
print("[{}] lenght_scale:= {}".format(arrow.now(), lenght_scale.numpy() ))
print("[{}] training.time:= {}".format(arrow.now(), training_time ))

loss_reward = Metrics(batch_size = 1000, random_state = rng).negative_rewards(expert_data, variance = variance, length_scale = length_scale )
loss_logl = Metrics(batch_size = 1000, random_state = rng).log_likelihood(expert_data, variance = variance, length_scale = length_scale )

print("IRL.Loss")
print("[{}] negative.reward.term : {}".format(arrow.now(), loss_reward ))
print("[{}] negative.log.likelihood : {}".format(arrow.now(), loss_logl))
print("[{}] time : {}".format(arrow.now(), time.time() - t0))