import numpy as np
import pickle


config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 1000  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.0001 # Learning rate of gradient descent algorithm

def softmax(x):
    """
    Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
    """
    output = np.array([np.exp(i)/np.sum(np.exp(i))for i in x])
    return output


def load_data(fname):
    """
    Write code to read the data and return it as 2 numpy arrays.
    Make sure to convert labels to one hot encoded format.
    """
    images = []
    labels = []
    data = pickle.load(open(fname, 'rb'), encoding='latin1')
    for d in data:
        images += [d[:-1]]
        labels += [[int(i == d[-1]) for i in range(10)]]
    images = np.array(images)
    labels = np.array(labels)
    return images, labels


class Activation:
    def __init__(self, activation_type = "sigmoid"):
        self.activation_type = activation_type
        self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
    def forward_pass(self, a):
        if self.activation_type == "sigmoid":
            return self.sigmoid(a)
    
        elif self.activation_type == "tanh":
            return self.tanh(a)
    
        elif self.activation_type == "ReLU":
            return self.ReLU(a)
  
    def backward_pass(self, delta):
        if self.activation_type == "sigmoid":
            grad = self.grad_sigmoid()
    
        elif self.activation_type == "tanh":
            grad = self.grad_tanh()
    
        elif self.activation_type == "ReLU":
            grad = self.grad_ReLU()
    
        return grad * delta
          
    def sigmoid(self, x):
        """
        Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = 1/(1+np.exp(-x))
        return output
    
    def tanh(self, x):
        """
        Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = np.tanh(x)
        return output

    def ReLU(self, x):
        """
        Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        self.x = x
        output = np.maximum(0,x)
        return output

    def grad_sigmoid(self):
        """
        Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
        """
        grad = np.exp(-self.x)/np.power(1+np.exp(-self.x),2) 
        return grad
    
    def grad_tanh(self):
        """
        Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
        """
        grad = 1/np.power(np.cosh(self.x),2)
        return grad

    def grad_ReLU(self):
        """
        Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
        """
        grad = np.copy(self.x)
        grad[grad <= 0] = 0
        grad[grad > 0] = 1
        return grad
    

class Layer():
    def __init__(self, in_units, out_units):
        np.random.seed(42)
        self.w = np.random.randn(in_units, out_units)  # Weight matrix
        self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
        self.x = None  # Save the input to forward_pass in this
        self.a = None  # Save the output of forward pass in this (without activation)
        self.d_x = None  # Save the gradient w.r.t x in this
        self.d_w = None  # Save the gradient w.r.t w in this
        self.d_b = None  # Save the gradient w.r.t b in this

    def forward_pass(self, x):
        """
        Write the code for forward pass through a layer. Do not apply activation function here.
        """
        self.x = x
        self.a = self.x @ self.w + np.repeat(self.b, len(x), axis = 0)
        return self.a
  
    def backward_pass(self, delta):
        """
        Write the code for backward pass. This takes in gradient from its next layer as input,
        computes gradient for its weights and the delta to pass to its previous layers.
        """
        self.d_x = delta @ self.w.T
        self.d_w = self.x.T @ delta 
        self.d_b = np.sum(delta,axis = 0 )
        return self.d_x

      
class Neuralnetwork():
    def __init__(self, config):
        self.layers = []
        self.x = None  # Save the input to forward_pass in this
        self.y = None  # Save the output vector of model in this
        self.targets = None  # Save the targets in forward_pass in this variable
        for i in range(len(config['layer_specs']) - 1):
            self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
            if i < len(config['layer_specs']) - 2:
                self.layers.append(Activation(config['activation']))  
    
    def forward_pass(self, x, targets=None):
        """
        Write the code for forward pass through all layers of the model and return loss and predictions.
        If targets == None, loss should be None. If not, then return the loss computed.
        """
        self.x = x
        self.targets = targets
        for l in self.layers:
            x = l.forward_pass(x)
            
        self.y = softmax(x)
        
        loss = self.loss_func(self.y, targets) if targets is not None else None
        return loss, self.y

    def loss_func(self, logits, targets):
        '''
        find cross entropy loss between logits and targets
        '''
        output = -np.sum(targets * np.log(logits))
        return output
    
    def backward_pass(self):
        '''
        implement the backward pass for the whole network. 
        hint - use previously built functions.
        '''
        delta = self.targets - self.y
        for l in self.layers[::-1]:
            delta = l.backward_pass(delta)
            
        return delta

    def trainer(model, X_train, y_train, X_valid, y_valid, config):
        """
        Write the code to train the network. Use values from config to set parameters
        such as L2 penalty, number of epochs, momentum, etc.
        """
  
  
    def test(model, X_test, y_test, config):
        """
        Write code to run the model on the data passed as input and return accuracy.
        """
        return accuracy
      
if __name__ == "__main__":
    train_data_fname = 'MNIST_train.pkl'
    valid_data_fname = 'MNIST_valid.pkl'
    test_data_fname = 'MNIST_test.pkl'
  
    ### Train the network ###
    model = Neuralnetwork(config)
    X_train, y_train = load_data(train_data_fname)
#     X_valid, y_valid = load_data(valid_data_fname)
#     X_test, y_test = load_data(test_data_fname)
#     trainer(model, X_train, y_train, X_valid, y_valid, config)
#     test_acc = test(model, X_test, y_test, config)