#!/bin/bash

if [[ "$1" = "load" ]]; then

    python3 convert.py --folder $2

fi

if [[ "$1" = "train" ]]; then

    python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=16,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=$3, predict_count=$3"  --train  ./datasets$2/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./datasets$2/tfrecords/evaluate/*.tfrecords --epochs 500 --model_dir ./CheckPoints/$2 

fi

if [[ "$1" = "predict" ]]; then

    python3 routenet.py --ini config.ini predict --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=50,predict_count=$3"  --predict ./datasets$2/tfrecords/evaluate/*.tfrecords --model_dir ./Model
fi

if [[ "$1" = "train_multiple" ]]; then

    for i in {1..50..2}
        do

<<<<<<< HEAD
        python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=$3, predict_count=$3"  --train  ./datasets$2/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./datasets/$2/tfrecords/evaluate/*.tfrecords --epochs 5 --model_dir ./CheckPoints/Model
        python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=$5, predict_count=$5"  --train  ./datasets$4/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./datasets/$4/tfrecords/evaluate/*.tfrecords --epochs 5 --model_dir ./CheckPoints/train_multiple
=======
        python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=$3, predict_count=$3"  --train  ./nsfnet/$2/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./nsfnet/$2/tfrecords/evaluate/*.tfrecords --epochs 5 --model_dir ./CheckPoints/Model
        python3 routenet.py --ini config.ini train --hparams="l2=0.1,dropout_rate=0.5,link_state_dim=32,path_state_dim=32,readout_units=256,learning_rate=0.001,T=8,node_count=$5, predict_count=$5"  --train  ./nsfnet/$4/tfrecords/train/*.tfrecords --train_steps 10 --eval_ ./nsfnet/$4/tfrecords/evaluate/*.tfrecords --epochs 5 --model_dir ./CheckPoints/Model
>>>>>>> origin/master
        
        done
fi

if [[ "$1" = "plots" ]]; then
    mkdir current_results
    # Results evaluated with 14 nodes
    python3 eager_eval.py --name dGlobal_0_8_AL_2_k_8 --routing Routing_AL_2_k_8 --nodes 14
    python3 eager_eval.py --name dGlobal_0_8_SP_k_0 --routing Routing_SP_k_0 --nodes 14
    python3 eager_eval.py --name dGlobal_0_10_AL_1_k_0 --routing Routing_AL_1_k_0 --nodes 14
    python3 eager_eval.py --name dGlobal_0_15_SP_k_58 --routing Routing_SP_k_58 --nodes 14

    # Results evaluated with 24 nodes
    python3 eager_eval.py --name dGlobal_G_0_8_AL_2_k_0 --routing RoutingGeant2_AL_2_k_0 --nodes 24
    python3 eager_eval.py --name dGlobal_G_0_8_SP_k_17 --routing RoutingGeant2_SP_k_17 --nodes 24

    # Results evaluated with 50 nodes
    python3 eager_eval.py --name dGlobal_0_8_SP_k_0 --routing SP_k_0 --nodes 50
    python3 eager_eval.py --name dGlobal_0_8_SP_k_89 --routing SP_k_89 --nodes 50
    python3 eager_eval.py --name dGlobal_0_8_W_4_k_3 --routing W_4_k_3 --nodes 50
    python3 eager_eval.py --name dGlobal_0_10_SP_k_0 --routing SP_k_0 --nodes 50
    python3 eager_eval.py --name dGlobal_0_10_SP_k_89 --routing SP_k_89 --nodes 50
    python3 eager_eval.py --name dGlobal_0_10_W_4_k_3 --routing W_4_k_3 --nodes 50
    python3 eager_eval.py --name dGlobal_0_12_SP_k_1 --routing SP_k_1 --nodes 50
    python3 eager_eval.py --name dGlobal_0_12_SP_k_89 --routing SP_k_89 --nodes 50
    python3 eager_eval.py --name dGlobal_0_12_W_4_k_3 --routing W_4_k_3 --nodes 50

    python3 cdf.py
fi