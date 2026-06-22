'''Modul opisujacy klase wartswa jako najmniejszej jednostki w mojej implementacji'''
import math
import random
from typing import Optional, Dict, Any
'''Ogolna klasa warstwy ktora to odpowiada za wagi,biasy oraz jej propagacje(przod/wstecz)'''
class Layer:
    def forward(self, inputs):
        raise NotImplementedError
        
    def backward(self, gradient_output):
        raise NotImplementedError
        
    def update_weights(self, step, momentum):
        pass

    def get_state(self) -> Optional[Dict[str, Any]]:
        return None
        
    def predefine(self, state: Dict[str, Any]):
        pass


'''Wzgledem tej klasy sa robione wszystkie 3 wartswy, gdyz jak uzyjemy je po 2 rozne strony od wartswy z aktywacja
To mozna zsymulowac to ze mamy iles wejsc i iles wyjsc przy uzyciu naszych macierzy'''
class Linear(Layer):
    def __init__(self, in_amount, out_amount, weight_range=0.5):
        self.in_amount = in_amount
        self.out_amount = out_amount
        self.weights = [[random.uniform(-weight_range, weight_range) for _ in range(out_amount)] for _ in range(in_amount)]
        self.biases = [0.0 for _ in range(out_amount)]
        self.speed_weights = [[0.0 for _ in range(out_amount)] for _ in range(in_amount)]
        self.speed_biases = [0.0 for _ in range(out_amount)]
        self.inputs = None
        self.gradient_weights = [[0.0 for _ in range(out_amount)] for _ in range(in_amount)]
        self.gradient_biases = [0.0 for _ in range(out_amount)]

    def get_state(self) -> Optional[Dict[str, Any]]:
        return {
            'weights': [[w for w in row] for row in self.weights],
            'biases': [b for b in self.biases]
        }

    def predefine(self, state: Dict[str, Any]):
        if state is None:
            return
        self.weights = state['weights']
        self.biases = state['biases']
        self.speed_weights = [[0.0 for _ in range(self.out_amount)] for _ in range(self.in_amount)]
        self.speed_biases = [0.0 for _ in range(self.out_amount)]

    def forward(self, inputs):
        self.inputs = inputs
        outputs = [0.0] * self.out_amount
        for j in range(self.out_amount):
            suma = self.biases[j]
            for i in range(self.in_amount):
                suma += self.inputs[i] * self.weights[i][j]
            outputs[j] = suma 
        return outputs
    
    def backward(self, gradient_output):
        gradient_input = [0.0] * self.in_amount
        for i in range(self.in_amount):
            suma = 0.0
            for j in range(self.out_amount):
                suma += gradient_output[j] * self.weights[i][j] # wlasciwe przekazanie bledu do wartswy wyzej
            gradient_input[i] = suma
            
        for j in range(self.out_amount):
            self.gradient_biases[j] = gradient_output[j]
            
        for i in range(self.in_amount):
            for j in range(self.out_amount):
                self.gradient_weights[i][j] = (gradient_output[j] * self.inputs[i]) if self.inputs is not None else 0
                
        return gradient_input

    def update_weights(self, step, momentum):
        for j in range(self.out_amount):
            self.speed_biases[j] = momentum * self.speed_biases[j] + step * self.gradient_biases[j]
            self.biases[j] += self.speed_biases[j]
            
        for i in range(self.in_amount):
            for j in range(self.out_amount):
                self.speed_weights[i][j] = momentum * self.speed_weights[i][j] + step * self.gradient_weights[i][j]
                self.weights[i][j] += self.speed_weights[i][j]

class Sigmoid(Layer):
    def forward(self, inputs):
        self.outputs = []
        for x in inputs:
            z = max(-250.0, min(x, 250.0))
            self.outputs.append(1.0 / (1.0 + math.exp(-z)))
        return self.outputs

    def backward(self, gradient_output):
        return [gradient_output[i] * self.outputs[i] * (1.0 - self.outputs[i]) for i in range(len(gradient_output))]

class Tanh(Layer):
    def forward(self, inputs):
        self.outputs = [math.tanh(x) for x in inputs]
        return self.outputs

    def backward(self, gradient_output):
        return [gradient_output[i] * (1.0 - self.outputs[i]**2) for i in range(len(gradient_output))]

class ReLU(Layer):
    def forward(self, inputs):
        self.inputs = inputs
        return [max(0.0, x) for x in inputs]

    def backward(self, gradient_output):
        return [gradient_output[i] * (1.0 if self.inputs[i] > 0.0 else 0.0) for i in range(len(gradient_output))]