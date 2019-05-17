
# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import routenet as upc
import configparser

tfe = tf.contrib.eager
tf.enable_eager_execution()

config = configparser.ConfigParser()
config.optionxform = str  # Disable lowercase conversion
config.read('config.ini')
normalization = config['Normalization']
delay_mean = float(normalization.get('mean_delay', 2.8))
traffic_mean = float(normalization.get('mean_traffic', 0.5))
delay_std = float(normalization.get('std_delay', 2.5))
traffic_std = float(normalization.get('std_traffic', .5))


hparams = upc.hparams.parse(
    "l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8, node_count=24")

model = upc.ComnetModel(hparams)
model.build()

saver = tfe.Saver(model.variables)
saver.restore('Model/model.ckpt-31008')

# 14 Nodes

R = upc.load_routing('nsfnet/14_nodes/routing/Routing_SP_k_0.txt')
con, n = upc.ned2lists('nsfnet/14_nodes/Network_14_nodes.ned')
paths = upc.make_paths(R, con)
link_indices, path_indices, sequ_indices = upc.make_indices(paths)

Global, TM_index, delay_index = upc.load(
    'nsfnet/14_nodes/delays/dGlobal_0_8_SP_k_0.txt', n)

delay = Global.take(delay_index, axis=1).values
TMs = Global.take(TM_index, axis=1).values
n_paths = delay.shape[1]
n_links = max(max(paths)) + 1
n_total = len(path_indices)
i = 344  # Grab just one traffic matrix

tm = tf.convert_to_tensor(
    (TMs[i, :] - traffic_mean) / traffic_std, dtype=tf.float32)

feature = {
    'traffic': tf.convert_to_tensor(tm, dtype=tf.float32),
    'links': link_indices,
    'paths': path_indices,
    'sequances': sequ_indices,
    'n_links': n_links,
    'n_paths': n_paths,
    'n_total': n_total
}

hats = [model(feature, training=True).numpy() for i in range(50)]

hats = delay_std * np.concatenate(hats, axis=1) + delay_mean

prediction = np.median(hats, axis=1)

x = prediction
y = delay[i, :]

relative_error = (y - x) / y

plt.hist(relative_error, cumulative=True, label='14 Nodes',
         histtype='step',bins=100, alpha=0.8, color='red', density=True)



# 24 Nodes



R = upc.load_routing('nsfnet/24_nodes/routing/RoutingGeant2_W_2_k_7.txt')
con, n = upc.ned2lists('nsfnet/24_nodes/Network_24_nodes.ned')
paths = upc.make_paths(R, con)
link_indices, path_indices, sequ_indices = upc.make_indices(paths)

Global, TM_index, delay_index = upc.load(
    'nsfnet/24_nodes/delays/dGlobal_G_0_12_W_2_k_7.txt', n)

delay = Global.take(delay_index, axis=1).values
TMs = Global.take(TM_index, axis=1).values
n_paths = delay.shape[1]
n_links = max(max(paths)) + 1
n_total = len(path_indices)
i = 344  # Grab just one traffic matrix

tm = tf.convert_to_tensor(
    (TMs[i, :] - traffic_mean) / traffic_std, dtype=tf.float32)

feature = {
    'traffic': tf.convert_to_tensor(tm, dtype=tf.float32),
    'links': link_indices,
    'paths': path_indices,
    'sequances': sequ_indices,
    'n_links': n_links,
    'n_paths': n_paths,
    'n_total': n_total
}

hats = [model(feature, training=True).numpy() for i in range(50)]

hats = delay_std * np.concatenate(hats, axis=1) + delay_mean

prediction = np.median(hats, axis=1)

x = prediction
y = delay[i, :]

relative_error = (y - x) / y

plt.hist(relative_error, cumulative=True, label='24 Nodes',
         histtype='step',bins=100, alpha=0.8, color='blue', density=True)


# 50 Nodes



R = upc.load_routing('nsfnet/50_nodes/routing/SP_k_0.txt')
con, n = upc.ned2lists('nsfnet/50_nodes/Network_50_nodes.ned')
paths = upc.make_paths(R, con)
link_indices, path_indices, sequ_indices = upc.make_indices(paths)

Global, TM_index, delay_index = upc.load(
    'nsfnet/50_nodes/delays/dGlobal_0_8_SP_k_0.txt', n)

delay = Global.take(delay_index, axis=1).values
TMs = Global.take(TM_index, axis=1).values
n_paths = delay.shape[1]
n_links = max(max(paths)) + 1
n_total = len(path_indices)
i = 344  # Grab just one traffic matrix

tm = tf.convert_to_tensor(
    (TMs[i, :] - traffic_mean) / traffic_std, dtype=tf.float32)

feature = {
    'traffic': tf.convert_to_tensor(tm, dtype=tf.float32),
    'links': link_indices,
    'paths': path_indices,
    'sequances': sequ_indices,
    'n_links': n_links,
    'n_paths': n_paths,
    'n_total': n_total
}

hats = [model(feature, training=True).numpy() for i in range(50)]

hats = delay_std * np.concatenate(hats, axis=1) + delay_mean

prediction = np.median(hats, axis=1)

x = prediction
y = delay[i, :]

relative_error = (y - x) / y

plt.hist(relative_error, cumulative=True, label='50 Nodes',
         histtype='step',bins=100, alpha=0.8, color='green', density=True)


plt.legend(prop={'size': 10})
plt.show()
