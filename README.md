# Java Neural Network
Like the name says, this is a minimal neural network (and I mean really minimal; no convolution, no validation during training, manually-tweaked hyperparameters, etc) written from scratch entirely in Java. It is trained to recognize handwritten digits on the [MNIST dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) which includes 60000 training samples and 10000 testing samples. The highest accuracy I've seen so far was 83.2%, which can probably be improved by introducing more complexity to the model but that is obviously far from practical in Java.

## Network Architecture
The inputs are 28x28 images (784 features) representing handwritten digits, where the value of each feature ranges from 0 to 255 representing pixel intensity. The network has 2 hidden layers, the first of which has 400 perceptrons and the second of which has 50. The activation function used in the hidden layers is [ReLU](https://en.wikipedia.org/wiki/ReLU), and for the output layer I used the [sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function). 

## But just... Why?
Obviously I'm aware Java is *far* from optimal for something like this. But I was required to take MET CS342 (Data Structures with Java) as a prerequisite course, so I needed a project to quickly familiarize myself with the language. So I was inspired by [this video](https://www.youtube.com/watch?v=ReOxVMxS83o) about building a neural network in C, and decided to also create my own neural network from scratch in an unconventional language to both learn that language and exercise my machine learning skills. Two birds, one stone kinda thing.

## Usage
```java
import Model.*;

// ...

Layer[] layers = new Layer[2];  // layer count
l[0] = new Layer(neuronCount, ActivationFunction.SIGMOID, LossFunction.LOGLOSS);
l[1] = new Layer(...); // so on
        
Model model = new Model(layers);

// skip this next line for binary classification or regression
model.setClasses(classCount);

model.train(x, y);

// make some prediction
double[] prediction = model.predict(xTest);
```
