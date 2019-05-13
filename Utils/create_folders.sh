#!/bin/bash
# Creates the setup for working with a new topology

DIR="$1_nodes"
mkdir nsfnet/$DIR
mkdir nsfnet/$DIR/delaysNsfnet
mkdir nsfnet/$DIR/routingNsfnet
mkdir nsfnet/$DIR/tfrecords
mkdir nsfnet/$DIR/tfrecords/train
mkdir nsfnet/$DIR/tfrecords/evalutate

echo "Make sure to copy the delays files and the routing files into the folders"
