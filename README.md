# RouteNet

RouteNet is a GNN implementation that aims to predict the delay and/or jitter of a network topology. Previous attempts where able to predict the delay of topologies with an amount of nodes close to the original training topology. With this new implementation we are able to train multiple topologies at the same time to create a GNN that is able to **generalize** the prediction of delay with topologies between 10 and 50 nodes. 

In this case we used 14 and 50 nodes as training for the GNN and 24 nodes as evaluation, achieving proimising results with the evaluations on the 3 topologies. It is also clear that RouteNet could also be used with more topologies, ej: 14, 50, 100, 150 nodes, to create a bigger generalization.

# SetUp
In order to setup the envoriment to try and run **RouteNet** we first need to install some python packages through *pip*
```sh
$ pip install -r Utils/requirements.txt
```
### File structure
The only important file structure is regarding the datasets, the datasets must be stored in a folder called nsfnet. So in this case the dataset structure would look like this.
  - root_folder
  - - nsfnet
  - - - 14_nodes
  - - - 24_nodes
  - - - 50_nodes 

In the root folder the rest of files would be stored. In the **Utils** folder there are a few scripts to help working or evaluating RouteNet.
  - normalize.py is used to create a config.ini file with the values to use to normalize the data in routenet.
  - eager_eval.py is used to create plots with the results comparing the true delay vs the predicted results.
  - create_folder.sh is used to create the file structure if a new topology is created.
# Running RouteNet
RouteNet can be run using the **run.sh** script. This script contains eases the ussage of the routenet.py, whihc is the actual file containing the routenet code.

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

Once the trainning procedure has been completed, tensorflow automatically creates a folder called **Checkpoints**, which contains the model trained, it also allows to check the metrics and other things about the model by using the *tensorboard* utility.
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

