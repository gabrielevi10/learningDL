import numpy as np 
import time

class NeuralNetwork():
    def __init__(self):
        np.random.seed(1)

        self.number_of_layers = 3 # a strange bug occurs with 10+ layers :/
        self.synaptic_weights = []
        
        self.synaptic_weights.append(2 * np.random.random((3, 4)) - 1)
        for i in range(self.number_of_layers - 3):
            self.synaptic_weights.append(2 * np.random.random((4, 4)) - 1)
        self.synaptic_weights.append(2 * np.random.random((4, 1)) - 1)

    def sigmoidFunction(self, x, derivative=False):
        if derivative:
            return(x * (1 - x))
        return(1/(1 + np.exp(-x)))

    def train(self, input, output, iterations):
        layer = 0
        layers_results = [[]] * self.number_of_layers
        layers_delta = [np.array([])] * self.number_of_layers

        for i in range(iterations):
            layers_results[0] = input
            layer += 1

            # feed forwarding
            for j in range(self.number_of_layers-1):
                layers_results[layer] = self.sigmoidFunction(np.dot(layers_results[j], self.synaptic_weights[j]))
                layer += 1 

            # print(self.synaptic_weights)   

            if layer % self.number_of_layers == 0:
                error = output - layers_results[layer-1]
                
                # backpropagation
                layers_delta[layer-1] = error * self.sigmoidFunction(layers_results[layer-1], derivative=True)
                for k in range(layer-2, -1, -1):
                    error = layers_delta[k+1].dot(self.synaptic_weights[k].T)
                    layers_delta[k] = error * self.sigmoidFunction(layers_results[k], derivative=True)                
                layer = 0
                
                for k in range(self.number_of_layers-1):
                    self.synaptic_weights[k] += layers_results[k].T.dot(layers_delta[k+1])

    def think(self, input):
        layer = 1
        layers_results = [0]*self.number_of_layers
        layers_results[0] = input
        for j in range(self.number_of_layers-1):
                layers_results[layer] = self.sigmoidFunction(np.dot(layers_results[j], self.synaptic_weights[j]))
                layer += 1
        return layers_results[layer-1]

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    trainning_input_dataset = np.array([[0, 0, 0],
                                        [1, 1, 1], 
                                        [1, 0, 1], 
                                        [0, 1, 1]])
    
    trainning_output_dataset = np.array([[0, 1, 1, 0]]).T

    neural_network.train(trainning_input_dataset, trainning_output_dataset, 60000)

    input_dataset = np.array([[0, 1, 0]])
                            
    print(neural_network.think(input_dataset))


    