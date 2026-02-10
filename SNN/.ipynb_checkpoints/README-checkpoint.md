# SNN
This directory contains all the files needed for the definition, training and testing of the Spiking Neural Network, and some auxiliary files for events visualization. The present document serves as a guide to understand the code, why it is this way and how to use it.

# Table of Contents

1. [Code Overview](#code-overview)
    - [SNN definition, training and testing](#snn-definition-training-and-testing)
        - [Spiking Network Class](#spiking-network-class)
        - [Predictor Class](#predictor-class)
        - [MultiLoss Class](#multiloss-class)
        - [Trainer Class](#trainer-class)
    - [Dataset Creation](#dataset-creation)
    - [Visualization](#visualization)
2. [How to Use](#how-to-use)
    - [Create a Dataset](#create-a-dataset)
    - [Define a Network](#Define-a-network)
3. [Future Developments](#future-developments)

---
# Code Overview

## SNN definition, training and testing
The code for this section is contained in the *SNN_func.py* file. Let's look at its different components.


### Spiking Network Class

```python
class Spiking_Net(nn.Module):
    """FCN with variable neural model and number of layers."""
```

This class, as the name suggests, is the one that implements the spiking neural network to be trained. It supports a Fully Connected (feedforward) Network with completely customizable parameters: number of layers, neurons per layer, neuron model and more. 

#### Initialization
The initialization function takes two input variables: *net_desc* and *spikegen_fn*.

```python
    def __init__(self, net_desc, spikegen_fn):
        super().__init__()
        
        self.n_neurons = net_desc["layers"]
        self.timesteps = net_desc["timesteps"]
        self.output = net_desc["output"]

        modules = []
        for i_layer in range(1, len(self.n_neurons)):
            modules.append(nn.Linear(in_features=self.n_neurons[i_layer-1], out_features=self.n_neurons[i_layer]))
            if "model" in net_desc:
                modules.append(net_desc["model"](**net_desc["neuron_params"][i_layer]))
            else:
                modules.append(net_desc["neuron_params"][i_layer][0](**(net_desc["neuron_params"][i_layer][1])))
        self.network = nn.Sequential(*modules)

        self.spikegen_fn = spikegen_fn
```
The first variable, *net_desc*, is a dictionary containing the necessary parameter information to build the network. In particular, it must define the following:
1. **layers**: a *list of int*, corresponding to the numbers of neurons in each layer of the network.
2. **timesteps**: an *int*, corresponding to how many timesteps are processed by the network.
3. **neuron_params**: a *dictionary*, where each entry specifies the neuron model of the corresponding layer. The keys must be *int*, starting from 1 (the input layer does not have a neuron mode, as it transmits the inputs from the afferents). The values are *lists*, where the first entry specififes *snntorch* neural model to use, while the second is itself a dictionary containing the named parameters to pass to the model initialization.
4. **output**: a *string* that must be equal either to *spike* or to *membrane* and defines what should be used as the output of the last layer of the network.
5. **model** (optional): if all layers use the same neural model, it can be specified using this keyword instead of repeated for all layers. In this case, the **neuron_patams** dictionary is simplified and its keys must only be the dictionaries containing the model parameters, not the lists.

The second input variable, *spikegen_fn*, is the function that converts the input data into a spiketrain to be fed to the network. The output of this function must be a **torch tensor** with binary values (i.e only 0s and 1s) of shape *[time, batch size, input layer dim]*. Note that time must be equal to the *timesteps* parameter defined in *net_desc*. <br>

Examples for both *net_desc* and *spikegen_fn* can be found in sub-section *[Define a Network](#define-a-network)*. <br>


#### Forward Step
The forward function is the one called when feeding the data to the network. 
```python
    def forward(self, data):
        """Forward pass for several time steps."""

        x = self.spikegen_fn(data)

        # Initalize membrane potential
        mem = []
        for i, module in enumerate(self.network):
            if i%2==1:
                res = module.reset_mem()
                if type(res) is tuple:
                    mem.append(list(res))
                else:
                    mem.append([res])

        # Record the final layer
        spk_rec = []
        mem_rec = []

        # Loop over 
        spk = None
        for step in range(self.timesteps):
            for i_layer in range(len(self.network)//2):
                if i_layer == 0:
                    cur = self.network[2*i_layer](x[step])
                else:
                    cur = self.network[2*i_layer](spk)
                
                spk, *(mem[i_layer]) = self.network[2*i_layer+1](cur, *(mem[i_layer]))

                if i_layer == len(self.network)//2-1:
                    spk_rec.append(spk)
                    mem_rec.append(mem[i_layer][-1])

        if self.output == "spike":
            return torch.stack(spk_rec, dim=0)
        elif self.output == "membrane":
            return torch.stack(mem_rec, dim=0)
```
It first transforms the input data into a spiketrain using *spikegen_fn* and then initializes the membrane of each layer. After that, it loops over all timesteps feeding the new spikes to the network and computing how the signal propagates through the layers. Finally, depending on the output selected in *net_desc*, the full history of the membrane/output spikes of the last layer are given in ouput (with shape *[time, batch size, output layer dim]*).

### Predictor Class

```python
class Predictor():
```
This simple class handles the conversion from network output to actual target prediction and computation of the prediction error.

#### Initialization

```python
    def __init__(self, predict, transform=None, relative=True, population_sizes=None):
        self.predict = predict
        self.transform = transform
        self.relative = relative
        self.population_sizes = population_sizes
```
#### Call
```python
    def _predict_singletask(self, output, targets):
        prediction = self.predict(output)
        if self.transform:
            prediction, targets = self.transform(prediction), self.transform(targets)
        accuracy = torch.abs(targets - prediction)
        if self.relative:
            accuracy /= targets
        return prediction, torch.mean(accuracy, 0)
    
    def __call__(self, output, targets):
        #check if multiple task must be handled
        if len(targets.shape) == 1:
            return self._predict_singletask(output, targets)
        
        # populations of different size
        if isinstance(self.population_sizes, (list, np.ndarray)):
            if sum(self.population_sizes) != output.shape[-1]:
                raise ValueError("Population sizes must add up to last layer size!")
            if len(self.population_sizes) != targets.shape[-1]:
                raise ValueError("Number of populations must be equal to number of tasks!")
            prediction = torch.zeros(size=targets.shape)
            accuracy = torch.zeros(size=(targets.shape[1],))
            chunks = list(accumulate([0]+list(self.population_sizes)))
            for i, pop in enumerate(self.population_sizes):
                prediction[:, i], accuracy[i] = \
                        self._predict_singletask(output[chunks[i]:chunks[i+1]], targets[:, i])
            return  prediction, accuracy          

        # populations of the same size
        if isinstance(self.population_sizes, int):
            output = output.reshape(output.shape[0], output.shape[1], self.population_sizes, -1)
            if output.shape[-1] != targets.shape[-1]:
                raise ValueError("Number of populations must be equal to number of tasks!")
        return self._predict_singletask(output, targets)

```



# How to Use

## Define a Network

Here is an example of how a *net_desc* object may look like:

```python
net_desc = {
    "layers" : [400, 50, 30],
    "timesteps": 100,
    "neuron_params" : {
                1: [snn.Leaky, 
                    {"beta" : 1.0,
                    "learn_beta": True,
                    "threshold" : 1.0,
                    "learn_threshold": True,
                    "spike_grad": surrogate.atan(),
                    }],
                2: [snn.Leaky, 
                    {"beta" : 1.0,
                    "learn_beta": True,
                    "threshold" : 1.0e20,
                    "learn_threshold": False, 
                    "spike_grad": surrogate.atan(),
                    }]
                },
    "output": "membrane"
}

```


Here is an example of this function:

```python
def spikegen_multi(data, multiplicity=4):
    og_shape = data.shape
    spike_data = torch.zeros(og_shape[1], og_shape[0], multiplicity*og_shape[2])
    for i in range(multiplicity):
        condition = data > np.power(10, i+2)
        batch_idx, time_idx, sensor_idx = torch.nonzero(condition, as_tuple=True)
        spike_data[time_idx, batch_idx, multiplicity*sensor_idx+i] = 1

    return spike_data
```
