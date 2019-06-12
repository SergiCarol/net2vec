# author:
# - 'Sergi Carol'

# coding: utf-8
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import routenet as upc
import configparser
import argparse
import random
<<<<<<< HEAD
=======
import glob
>>>>>>> origin/master


def calculate_true_delay(R, ned, data, i):
    R = upc.load_routing(R)
    con, n = upc.ned2lists(ned)
    paths = upc.make_paths(R, con)
    link_indices, path_indices, sequ_indices = upc.make_indices(paths)

    Global, TM_index, delay_index = upc.load(data, n)

    delay = Global.take(delay_index, axis=1).values
    TMs = Global.take(TM_index, axis=1).values
    n_paths = delay.shape[1]
    n_links = max(max(paths)) + 1
    n_total = len(path_indices)

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
    return (delay[i, :], feature)


def calculate_delay_from_model(feature):

    hats = [model(feature, training=True).numpy() for i in range(50)]

    hats = delay_std * np.concatenate(hats, axis=1) + delay_mean

    prediction = np.median(hats, axis=1)

    return prediction


def r_squared(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r_squared = 1 - (sum((y - (slope * x + intercept))**2) /
                     ((len(y) - 1) * np.var(y, ddof=1)))
    return r_squared


parser = argparse.ArgumentParser()
parser.add_argument("--file", help="Read result file and calculate from there",
                    type=str, required=False)

args = parser.parse_args()
try:
    res = np.load(args.file)
except ValueError:
    res = None

if res is not None:
    mre_14 = res['mre_14']
    mre_24 = res['mre_24']
    mre_50 = res['mre_50']
    r_2_14 = res['r_2_14']
    r_2_24 = res['r_2_24']
    r_2_50 = res['r_2_50']
    print('*' * 50)
    print('Results:')
    print('R^2')
    print("14 Nodes", r_2_14)
    print("24 Nodes", r_2_24)
    print("50 Nodes", r_2_50)
    print('*' * 50)

    plt.hist(mre_14, cumulative=True, label='14 Nodes',
             histtype='step', bins=100, alpha=0.8, color='blue', density=True)
    plt.hist(mre_24, cumulative=True, label='24 Nodes',
             histtype='step', bins=100, alpha=0.8, color='red', density=True)
    plt.hist(mre_50, cumulative=True, label='50 Nodes',
             histtype='step', bins=100, alpha=0.8, color='green', density=True)
    plt.title("CDF")

    plt.legend(prop={'size': 10})
    plt.show()

    exit()


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
saver.restore('CheckPoints/train_multiple/Training/model.ckpt-31008')

# 14 Nodes

<<<<<<< HEAD
routing = ['datasets/14_nodes/routing/Routing_SP_k_0.txt',
           'datasets/14_nodes/routing/Routing_AL_1_k_1.txt',
           'datasets/14_nodes/routing/Routing_SP_k_15.txt',
           'datasets/14_nodes/routing/Routing_SP_k_32.txt',
           'datasets/14_nodes/routing/Routing_W_8_k_9.txt',
           'datasets/14_nodes/routing/Routing_W_3_k_4.txt',
           'datasets/14_nodes/routing/Routing_SP_k_46.txt',
           'datasets/14_nodes/routing/Routing_SP_k_46.txt',
           'datasets/14_nodes/routing/Routing_SP_k_20.txt']

delays = ['datasets/14_nodes/delays/dGlobal_0_8_SP_k_0.txt',
          'datasets/14_nodes/delays/dGlobal_0_10_AL_1_k_1.txt',
          'datasets/14_nodes/delays/dGlobal_0_10_SP_k_15.txt',
          'datasets/14_nodes/delays/dGlobal_0_12_SP_k_32.txt',
          'datasets/14_nodes/delays/dGlobal_0_15_W_8_k_9.txt',
          'datasets/14_nodes/delays/dGlobal_0_10_W_3_k_4.txt',
          'datasets/14_nodes/delays/dGlobal_0_8_SP_k_46.txt',
          'datasets/14_nodes/delays/dGlobal_0_10_SP_k_46.txt',
          'datasets/14_nodes/delays/dGlobal_0_8_SP_k_20.txt']

mre = []
r_2 = []

for j in range(len(routing)):
    print("Using", routing[j], delays[j])
    for i in random.sample(range(1, 500), 25):
        y, feature = calculate_true_delay(upc.infer_routing_nsf(delays[j]),
                                          'datasets/14_nodes/Network_14_nodes.ned',
                                          delays[j], i)

        x = calculate_delay_from_model(feature)

        relative_error = (y - x) / y

        mre.append(relative_error)
        r_2.append(r_squared(x, y))

mre = np.asarray(mre)
avg_mre_14 = np.average(mre, axis=0)
r_2_14 = np.average(r_2)

=======
evaluate_records = random.sample(
    glob.glob('nsfnet/14_nodes/tfrecords/evaluate/*.tfrecords'), 100)
delays = []
path = 'nsfnet/14_nodes/delays/%s.txt'
for record in evaluate_records:
    tfrecord = record.split('/')[-1]
    file = tfrecord.split('.')[0]
    delays.append(path % file)


mre = []
r_2 = []

for j in range(len(delays)):
    print("Using", upc.infer_routing_nsf(delays[j]), delays[j])
    for i in random.sample(range(1, 500), 30):
        try:
            y, feature = calculate_true_delay(upc.infer_routing_nsf(delays[j]),
                                              'nsfnet/14_nodes/Network_14_nodes.ned',
                                              delays[j], i)

            x = calculate_delay_from_model(feature)

            relative_error = (y - x) / y

            mre.append(relative_error)
            r_2.append(r_squared(x, y))
        except:
            continue

mre = np.asarray(mre)
avg_mre_14 = np.average(mre, axis=0)
r_2_14 = np.average(r_2)

>>>>>>> origin/master
print("R Squared", r_2_14)

plt.hist(avg_mre_14, cumulative=True, label='14 Nodes',
         histtype='step', bins=100, alpha=0.8, color='blue', density=True)


# 24 Nodes
delays = random.sample(glob.glob('nsfnet/24_nodes/delays/*.txt'), 100)

<<<<<<< HEAD
routing = ['datasets/24_nodes/routing/RoutingGeant2_W_2_k_7.txt',
           'datasets/24_nodes/routing/RoutingGeant2_AL_2_k_0.txt',
           'datasets/24_nodes/routing/RoutingGeant2_SP_k_0.txt',
           'datasets/24_nodes/routing/RoutingGeant2_SP_k_51.txt',
           'datasets/24_nodes/routing/RoutingGeant2_SP_k_59.txt',
           'datasets/24_nodes/routing/RoutingGeant2_W_6_k_7.txt',
           'datasets/24_nodes/routing/RoutingGeant2_AL_2_k_8.txt',
           'datasets/24_nodes/routing/RoutingGeant2_SP_k_62.txt',
           'datasets/24_nodes/routing/RoutingGeant2_W_2_k_4.txt']

delays = ['datasets/24_nodes/delays/dGlobal_G_0_12_W_2_k_7.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_12_AL_2_k_0.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_10_SP_k_0.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_15_SP_k_0.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_12_SP_k_59.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_12_W_6_k_7.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_10_AL_2_k_8.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_8_SP_k_62.txt',
          'datasets/24_nodes/delays/dGlobal_G_0_15_W_2_k_4.txt']

mre = []
r_2 = []

for j in range(len(routing)):
    print("Using", routing[j], delays[j])
    for i in random.sample(range(1, 500), 25):
        y, feature = calculate_true_delay(routing[j],
                                          'datasets/24_nodes/Network_24_nodes.ned',
                                          delays[j], i)

        x = calculate_delay_from_model(feature)

        relative_error = (y - x) / y

        mre.append(relative_error)
        r_2.append(r_squared(x, y))

mre = np.asarray(mre)
avg_mre_24 = np.average(mre, axis=0)
r_2_24 = np.average(r_2)

print("R Squared", r_2_24)

=======
mre = []
r_2 = []

for j in range(len(delays)):
    print("Using", upc.infer_routing_geant(delays[j]), delays[j])
    for i in random.sample(range(1, 500), 30):
        try:    
            y, feature = calculate_true_delay(upc.infer_routing_geant(delays[j]),
                                              'nsfnet/24_nodes/Network_24_nodes.ned',
                                              delays[j], i)

            x = calculate_delay_from_model(feature)

            relative_error = (y - x) / y

            mre.append(relative_error)
            r_2.append(r_squared(x, y))
        except:
            continue
mre = np.asarray(mre)
avg_mre_24 = np.average(mre, axis=0)
r_2_24 = np.average(r_2)

print("R Squared", r_2_24)

>>>>>>> origin/master
plt.hist(avg_mre_24, cumulative=True, label='24 Nodes',
         histtype='step', bins=100, alpha=0.8, color='red', density=True)


# 50 Nodes

<<<<<<< HEAD
routing = ['datasets/50_nodes/routing/SP_k_0.txt',
           'datasets/50_nodes/routing/SP_k_21.txt',
           'datasets/50_nodes/routing/SP_k_61.txt',
           'datasets/50_nodes/routing/W_6_k_9.txt',
           'datasets/50_nodes/routing/SP_k_90.txt',
           'datasets/50_nodes/routing/W_10_k_0.txt',
           'datasets/50_nodes/routing/W_1_k_9.txt',
           'datasets/50_nodes/routing/SP_k_4.txt',
           'datasets/50_nodes/routing/W_3_k_3.txt']
=======
evaluate_records = random.sample(
    glob.glob('nsfnet/50_nodes/tfrecords/evaluate/*.tfrecords'), 100)
delays = []
path = 'nsfnet/50_nodes/delays/%s.txt'
for record in evaluate_records:
    tfrecord = record.split('/')[-1]
    file = tfrecord.split('.')[0]
    delays.append(path % file)
>>>>>>> origin/master

delays = ['datasets/50_nodes/delays/dGlobal_0_8_SP_k_0.txt',
          'datasets/50_nodes/delays/dGlobal_0_8_SP_k_21.txt',
          'datasets/50_nodes/delays/dGlobal_0_10_SP_k_61.txt',
          'datasets/50_nodes/delays/dGlobal_0_10_W_6_k_9.txt',
          'datasets/50_nodes/delays/dGlobal_0_12_SP_k_90.txt',
          'datasets/50_nodes/delays/dGlobal_0_12_W_10_k_0.txt',
          'datasets/50_nodes/delays/dGlobal_0_8_W_1_k_9.txt',
          'datasets/50_nodes/delays/dGlobal_0_12_SP_k_4.txt',
          'datasets/50_nodes/delays/dGlobal_0_12_W_3_k_3.txt']

mre = []
r_2 = []
<<<<<<< HEAD

for j in range(len(routing)):
    print("Using", routing[j], delays[j])
    for i in random.sample(range(1, 500), 25):
        y, feature = calculate_true_delay(routing[j],
                                          'datasets/50_nodes/Network_50_nodes.ned',
                                          delays[j], i)

        x = calculate_delay_from_model(feature)

        relative_error = (y - x) / y

        mre.append(relative_error)
        r_2.append(r_squared(x, y))

=======

for j in range(len(delays)):
    print("Using", upc.infer_routing_nsf2(delays[j]), delays[j])
    for i in random.sample(range(1, 500), 30):
        try:
            y, feature = calculate_true_delay(upc.infer_routing_nsf2(delays[j]),
                                              'nsfnet/50_nodes/Network_50_nodes.ned',
                                              delays[j], i)

            x = calculate_delay_from_model(feature)

            relative_error = (y - x) / y

            mre.append(relative_error)
            r_2.append(r_squared(x, y))
        except:
            continue
>>>>>>> origin/master
mre = np.asarray(mre)
avg_mre_50 = np.average(mre, axis=0)
r_2_50 = np.average(r_2)

print("R Squared", r_2_50)
plt.hist(avg_mre_50, cumulative=True, label='50 Nodes',
         histtype='step', bins=100, alpha=0.8, color='green', density=True)


np.savez(args.file, mre_14=avg_mre_14, mre_24=avg_mre_24, mre_50=avg_mre_50,
         r_2_14=r_2_14, r_2_24=r_2_24, r_2_50=r_2_50)

plt.title("CDF")

plt.legend(prop={'size': 10})
plt.show()
