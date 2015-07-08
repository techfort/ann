function rand() {
    return (Math.random() * 2.0) - 1;
}

var util = {};
util.randn = function (rows, cols) {
    var matrix = [],
        row;
    for (var r = 0; r < rows; r += 1) {
        row = [];
        for (var c = 0; c < cols; c += 1) {
            row.push(rand());
        }
        matrix.push(row);
    }
    return matrix;
};

util.argmax = function (arr) {
    return arr.indexOf(Math.max.apply(Math, arr));
};

util.xrange = function (min, max) {

}

util.vectorize = function (func) {
    return function (array, val) {
        var result = [];
        array.forEach(function (elem) {
            result.push(func(elem, val));
        });
        return result;
    }
};

util.sigmoid = function (x) {
    return 1.0 / (1.0 + Math.exp(-x));
};

util.sigmoid_vec = util.vectorize(util.sigmoid);

util.sigmoid_prime = function (z) {
    return util.sigmoid(z) * (1 - util.sigmoid(z));
};

util.sigmoid_prime_vec = util.vectorize(util.sigmoid_prime);

util.zip = function (a, b) {
    if (a.length !== b.length) {
        throw new Error('zipping requires same length arrays');
    }
    var result = [],
        i = 0,
        len = a.length;
    for (i; i < len; i += 1) {
        result.push([a[i], b[i]]);
    }
    return result;
}

util.dot = function (a, b) {
    var n = 0,
        lim = Math.min(a.length, b.length);
    for (var i = 0; i < lim; i++) n += a[i] * b[i];
    return n;
};

util.zeros = function (shape) {
    if (typeof shape === 'number') {
        var array = nnArray(shape);
        array.fill(0);
        return array;
    }

    var rows = shape[0],
        cols = shape[1],
        array = [];

    util.for(rows, function () {
        var row = nnArray(cols);
        row.fill(0);
        array.push(row);
    });

    return array;
};

util.for = function (times, func) {
    var i = 0;
    for (i; i < times; i += 1) {
        func();
    }
};

util.shuffle = function (array) {
    var o = Object.create(array);
    for (var j, x, i = o.length; i; j = Math.floor(Math.random() * i),
        x = o[--i],
        o[i] = o[j],
        o[j] = x);
    return o;
};

function nnArray(arr) {
    var array;
    if (arr) {
        array = typeof arr === 'number' ? new Array(arr) : arr;
    } else {
        array = [];
    }

    Object.defineProperty(array, 'fill', {
        enumerable: false,
        writable: false,
        value: function (value) {
            var len = array.length,
                x = 0;
            for (x; x < len; x += 1) {
                array[x] = typeof value === 'function' ? value.apply(array[x]) : value;
            }
        }
    });

    Object.defineProperty(array, 'at', {
        enumerable: false,
        writable: false,
        value: function (num) {
            if (num > -1) {
                return array[num];
            }
            return array[array.length + num];
        }
    });
    return array;
}

function Network(sizesArray) {
    this.numLayers = sizesArray.length;
    this.sizes = sizesArray;

    // set the biases
    this.biases = (function () {
        var layers = Object.create(sizesArray);
        // take out the output layer
        layers.pop();
        // take out the input layer
        layers.shift();
        var biases = [];
        layers.forEach(function (len) {
            var array = nnArray(len);
            array.fill(rand);
            biases.push(array);
        });
        return biases;
    }());

    this.weights = (function () {
        var a = Object.create(sizesArray),
            b = Object.create(sizesArray),
            len = a.length,
            i = 0;
        // input to last hidden
        a.pop();
        // first hidden to output
        b.shift();

        var weights = [];
        util.zip(a, b).forEach(function (pair) {
            weights.push(util.randn(pair[0], pair[1]));
        });
        return weights;
    }());
}

Network.prototype.feedforward = function (a) {
    util.zip(this.biases, this.weights).forEach(function (pair) {
        var bias = pair[0],
            weight = pair[1];
        a = util.sigmoid_vec(util.dot(weight, a) + bias);
    });
    return a;
};

Network.prototype.SGD = function (training_data, epochs,
    mini_batch_size, eta, test_data) {

};

Network.prototype.updateMiniBatch = function (mini_batch, eta) {

};

Network.prototype.backprop = function () {

};

Network.prototype.evaluate = function (test_data) {

    var sum = 0,
        self = this;
    test_data.forEach(function (t) {
        sum += util.argmax(self.feedforward(t[0])) === t[1] ? 1 : 0;
    });
    return sum;
};
Network.prototype.costDerivative = function (output_activations, y) {
    return (output_activations - y);
};

var nn = new Network([2, 4, 3, 1]);
console.log(nn.biases);
console.log('weights', nn.weights);


/*
"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network():

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) 
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid_vec(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid_vec(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime_vec(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            spv = sigmoid_prime_vec(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * spv
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) 
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        
    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

sigmoid_vec = np.vectorize(sigmoid)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)
*/
