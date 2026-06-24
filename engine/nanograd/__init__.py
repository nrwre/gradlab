"""nanograd — a small from-scratch deep-learning + classical-ML engine.

Built on top of coursework in optimisation, regression, SVMs and tree
ensembles (IIT Kharagpur, EE22202). It provides:

  * a reverse-mode autograd core (Value)
  * first-order optimisers (SGD, Momentum, Nesterov, Adam)
  * tiny neural-net layers and losses
  * from-scratch classical models (RBF/linear SVM, decision tree, random forest)
  * benchmark loss surfaces and toy datasets

The point is not to replace PyTorch/scikit-learn but to demonstrate that the
underlying mathematics is genuinely understood.
"""
from .value import Value
from .optim import SGD, Momentum, Nesterov, Adam
from .nn import Linear, MLP, mse_loss, softmax_cross_entropy
from .problems import quadratic_surface, rosenbrock, make_blobs, make_moons, make_circles
from .svm import SVM
from .forest import DecisionTree, RandomForest

__all__ = [
    "Value", "SGD", "Momentum", "Nesterov", "Adam",
    "Linear", "MLP", "mse_loss", "softmax_cross_entropy",
    "quadratic_surface", "rosenbrock", "make_blobs", "make_moons", "make_circles",
    "SVM", "DecisionTree", "RandomForest",
]
__version__ = "0.1.0"
