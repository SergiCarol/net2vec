# RouteNet

RouteNet is a GNN implementation that aims to predict the delay and/or jitter of a network topology. Previous attempts were able to predict the delay of topologies with an amount of nodes close to the original training topology. With this new implementation we are able to train multiple topologies at the same time to create a GNN that is able to **generalize** the prediction of delay with topologies between 10 and 50 nodes. 

In this case we used 14 and 50 nodes as training for the GNN and 24 nodes as evaluation, achieving promising results with the evaluations on the 3 topologies. It is also clear that RouteNet could also be used with more topologies, ej: 14, 50, 100, 150 nodes, to create a bigger generalization.


# SetUp
In order to setup the environment to try and run **RouteNet** we first need to install some python packages through *pip*
```sh
$ pip install -r Utils/requirements.txt
```

### The DataSets

The three datasets used are for 14, 24 and 50 nodes and can be found here [DataSets](https://github.com/knowledgedefinednetworking/NetworkModelingDatasets/tree/master/datasets_v0)

### File structure
The only important file structure is regarding the datasets, the datasets must be stored in a folder called nsfnet. So in this case the dataset structure would look like this.
  - root_folder
  - - nsfnet
  - - - 14_nodes
  - - - 24_nodes
  - - - 50_nodes 

Each dataset folder (14_nodes, 24_nodes and 50_nodes) contains three folders, *delays*, *routings*, and *tfrecords*. delays and routings contains the dataset itself, while the tfrecords folders contains the pre processed data.

In the root folder the rest of files would be stored. In the **Utils** folder there are a few scripts to help working or evaluating RouteNet.
  - normalize.py is used to create a config.ini file with the values to use to normalize the data in routenet.
  - eager_eval.py is used to create plots with the results comparing the true delay vs the predicted results.
  - create_folder.sh is used to create the file structure if a new topology is created.

# Running RouteNet
RouteNet can be run using the **run.sh** script. This script contains eases the ussage of the routenet.py, which is the actual file containing the routenet code.

In order to train a new model from scratch we use:
```sh
$ ./run.sh train 14_nodes 14
```
Or
```sh
$ ./run.sh train_multiple 14_nodes 14 50_nodes 50
```
In all the cases, the first parameter is used to identify which action to take, the second is used to know which topology is going to be used, and the last is used to know the number of nodes in that topology. The main difference between *train* and *train_multiple* is that *train* is used only to train a model with one topology, and *train_multiple* is used with various topologies.

### Validation

Once the training procedure has been completed, tensorflow automatically creates a folder called **Checkpoints**, which contains the model trained, it also allows to check the metrics and other things about the model by using the *tensorboard* utility.
```sh
$ tensorboard --logdir=CheckPoints
```
In a web browser we check the results.

It is also possible to plot the results in the form of True Delay vs Predicted using the *run.sh* script, which will create a series of plots in a folder on the root directory called *current_results*
```sh
$ ./run.sh plots
```
Finally if we want to check how the metrics would behave in a topology we can use the predict utility. In the case below we would predict the topology with 24 nodes using the model stored in the train folder on the Checkpoints.
```sh
$ ./run.sh predict 24_nodes 24 train
```

### Using other topologies

The datasets provided are already pre processed, and the config file provided already contains the values needed to train RouteNet with the 14 and 50 topologies. But it can be the case that you wish to use a different topology than the ones provided, in this case there are two steps that need to be done before training.

We begin by converting the data from the simulations to *tfrecords* which is a data format to work with *tensorflow*. To do so we use the *run.sh* utility.

```sh
$ ./run.sh load 50_nodes
```
Which will automatically convert the data and split the dataset into training and evaluation with a proportion of 80% and 20% of the data.

Then we choosing the normalization values, this can be done using the *normalize.py*. An example of utilization would be the following, where we choose the topology to use and it will automatically calculate the appropriate values and store them in the *config.ini* file. 
```sh
$  python3 Utils/normalize.py --dir nsfnet/14_nodes/tfrecords/train --ini config.ini
```
We can also use it to normalize input from multiple sources at once.

```sh
$  python3 Utils/normalize.py --dir nsfnet/14_nodes/tfrecords/train nsfnet/50_nodes/tfrecords/train --ini config.ini
```

