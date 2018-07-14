
# PyTorch Primer

A collection of didactic examples of machine learning algorithms using Pytorch

### Nonlinear Function Approximation Using a Neural Network

This example uses a visualization tool to show the ability of a neural network to approximate a
nonlinear function of the form *y = f(x) + n*, where *n* is random observation noise. In the example,
the nonlinear function is:

      *y = x^4 tanh(x) + n*

where *n* is observation noise from a normal distribution with zero mean and a standard deviation of *0.2*.

The neural network uses *x* as the input (i.e. input dimension is *1*), and *y* as the output (i.e. output dimension
is also *1*). Between the input and output, there is a hidden layer of *20* nodes with nonlinear relu activations.

The program shows how we can use Stochastic Gradient Descent to train this neural network to perform function
approximation. The program visualization shows a timeline of progressive improvements from successive
iterations.

The program requires:

* Python 2.7
* Pytorch 0.4

Donwload the program to a local machine and run from terminal:

```
python nlapprox.py
```

### License

This project is licensed under the MIT License (Expat version)

Copyright (c) 2018 Edwin Heredia

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.

