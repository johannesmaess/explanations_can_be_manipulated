from .convolutional import Convolutional
from .dense import Dense
from .pooling import MaxPool, SumPool, Flatten

# manual reload 
del MaxPool, SumPool, Flatten, Convolutional
from .pooling import MaxPool, SumPool, Flatten
from .convolutional import Convolutional