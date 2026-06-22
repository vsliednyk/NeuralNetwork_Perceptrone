'''Implementacja sieci neuronowej jako klasy ktora ma w sobie liste obiektow wartsw 
oraz wszystkich narzedzi wymaganych'''
import random
from layer import Linear, Sigmoid, Tanh, ReLU

class NeuralNetwork:
    def __init__(self, in_amount=2, hidden_amount=10, out_amount=2, activation='tanh', step=0.01, momentum=0.9, weight_range=0.5):
        self.step = step
        self.momentum = momentum
        self.out_amount = out_amount 
        
        self.layers = []
        self.layers.append(Linear(in_amount, hidden_amount, weight_range))
        
        if activation == 'sigmoid':
            self.layers.append(Sigmoid())
        elif activation == 'tanh':
            self.layers.append(Tanh())
        elif activation == 'relu':
            self.layers.append(ReLU())
        else:
            raise ValueError("Unsupported activation function!")
            
        self.layers.append(Linear(hidden_amount, out_amount, weight_range))
        self.last_output = None

    def export_model(self):
        return [layer.get_state() for layer in self.layers]

    def import_model(self, model_states):
        for layer, state in zip(self.layers, model_states):
            if state is not None:
                layer.predefine(state)

    def forward(self, inputs):
        current_data = inputs
        for current_layer in self.layers:
            current_data = current_layer.forward(current_data)
        self.last_output = current_data
        return current_data

    def backward(self, target_outputs):
        current_gradient = [target_outputs[i] - self.last_output[i] for i in range(self.out_amount)]  # pyright: ignore[reportOptionalSubscript]
        for current_layer in reversed(self.layers):
            current_gradient = current_layer.backward(current_gradient)

    def update_weights(self):
        for current_layer in self.layers:
            current_layer.update_weights(self.step, self.momentum)

    def calculate_mse(self, predicted, target):
        return (predicted[0] - target[0])**2 + (predicted[1] - target[1])**2

    def train(self, train_x, train_y, test_x=None, test_y=None, epochs=100):
        history_train_mse = []
        history_test_mse = [] 
        amount_of_train_patterns = len(train_x)
        indexes = list(range(amount_of_train_patterns))
        
        for current_epoch in range(epochs):
            random.shuffle(indexes)
            running_train_mse_sum = 0.0
            
            for index in indexes:
                x = train_x[index]
                y = train_y[index]
                predictions = self.forward(x)
                running_train_mse_sum += self.calculate_mse(predictions, y)
                self.backward(y)
                self.update_weights()
                
            epoch_mse = running_train_mse_sum / (amount_of_train_patterns * 2.0)
            history_train_mse.append(epoch_mse)
            
            if test_x is not None and test_y is not None:
                epoch_test_mse = self.test(test_x, test_y)
                history_test_mse.append(epoch_test_mse)
            
        return history_train_mse, history_test_mse

    def test(self, test_x, test_y):
        amount_of_test_patterns = len(test_x)
        running_test_mse_sum = 0.0
        for i in range(amount_of_test_patterns):
            predictions = self.forward(test_x[i])
            running_test_mse_sum += self.calculate_mse(predictions, test_y[i])
        return running_test_mse_sum / (amount_of_test_patterns * 2.0)