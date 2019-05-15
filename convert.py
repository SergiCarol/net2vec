# author:
# - 'Sergi Carol'

import glob
import os
import argparse
import routenet
import random

parser = argparse.ArgumentParser(description='Convert to TF records')
parser.add_argument("--folder", help="Target Folder", required=True)
parser.add_argument("--isNew", help="New data format", default=False)

args = parser.parse_args()

target = args.folder

for fname in glob.glob('nsfnet/' + target + '/delays/*.txt'):
    print(fname)
    tfname = fname.replace('txt', 'tfrecords')
    routenet.make_tfrecord2(tfname,
                       'nsfnet/' + target + '/Network_' + target + '.ned',
                       routenet.infer_routing_nsf(fname),
                       fname,
                       args.isNew == 'True'
                       )

tfrecords = glob.glob('nsfnet/' + target + '/delays/*.tfrecords')
traning = len(tfrecords) * 0.8
train_samples = random.sample(tfrecords, int(traning))
evaluate_samples = list(set(tfrecords) - set(train_samples))

for file in train_samples:
    file_name = file.split('/')[-1]
    os.rename(file, 'nsfnet/' + target + '/tfrecords/train/' + file_name)


for file in evaluate_samples:
    file_name = file.split('/')[-1]
    os.rename(file, 'nsfnet/' + target + '/tfrecords/evaluate/' + file_name)