'''
This is the program I used to generate the neural network
regression graphs on the center of my poster. This program
relies on Keras, the deep learning api. The best way I
think, to enjoy this api is downloading Pycharm and
importing the library there. If you don't it is quite
difficult.

Created by:     Devon Hastings
Documented:     April 19, 2019
Class:          Mathematical Exposistions
Email:          dth1002@plymouth.edu

~ Feel free to contact me if anyone has questions ~
'''

import numpy as np
import matplotlib.pyplot as plt
from keras import models, layers, optimizers
from keras.optimizers import SGD





# ========= Resolution and Domain ========= #
'''
By changing the resolution your network your
graph will look more smooth. If you only care
about predictions over integer values set the
resolution equal to 1.

The domain is from the negative value to the
postive calue of whatever value you choose.
In my example I set the domain equal to 4.
So our input data set will be from -4 to 4
in incrememnts of 0.1. Which in this case
is 81 points of data.
'''
resolution = 0.1
domain = 4
# ========================================= #





# ================ Raw Data =============== #
'''
We must make an array that contains our
points of data and we do this using

            np.arange()

This takes a start and finish and a
resolution. We add the resolution to the
right end of our data set because
np.arange() does one resolution less than
what we input.

Our function to approximate can be any
function. In this case we choose x^2.
You can build any fucntion here following
standard Python notation for mathematical
operations.
'''
x = np.arange(-domain, domain + resolution, resolution)
function_to_approximate = x ** 2
# ========================================= #





# =========== Formatting Data ============= #
'''
The networks built using Keras cannot take
in the data straight from np.arange() since
the array provided by np.arange() is, in our
case, (81,) but we need (81,1). This is
easy using the reshape method that comes with
Numpy (The api we use to build
matirces/arrays).

Next we must create labels, or in other words,
the values the network is trying to learn. As
an example we are trying to map (4)^2 = 16. So,
4 is in our data and 16 is a label for 4. This
is built to have the same dimesions as our data
array.
'''
data = x.reshape(x.shape[0],1)
label = function_to_approximate.reshape(function_to_approximate.shape[0], 1)
# ========================================= #





# ================ The Neural Network ================= #
'''
Alright, so now we have the meat of the program, the
neural network. To design a netwok to take in our data
we must use the first line after

            model = models.Sequential()

which is actually our second layer in the network.
Notice that the first parameter is 2. The 2 means 2 nodes
in that layer. The next line then builds the last layer
which only has 1 node.

                    NETWORK VISUAL

                        (L2)
                       /    \
                    (X)      (L3)
                       \    /
                        (L2)

The activation parameter lets you choose what type of
preset activation function that comes with Keras. We
let the last layer be linear since we are performing
regression and having a sigmoid would squish the
output between 0 and 1. To add layers follow the
commented out line and change the 2 to another number.
copy and paste the code between the last layer and first
layer until you have somthing you desire.

The compile method lets use choose what metrics to
record and our learning rate. I suggest reading
more on gradient descent to get an intuition on what
this value can be interpreted as. In compile we also
choose our cost/loss function. In this case we use

                Mean Squared Error/Loss

in order to keep this as basic and intuitive as
possible. This is also know as the "L2 norm".

In fit we choose our epoch size and batch size,
but we choose the batch size to be the whole
data set. Since the batch size is the whole data
set our cost function contains the amount of terms
equal to the length of our data size. This helps
each epoch have the best possible change of
weights during an epoch. Set verbose to true
to monitor training during each epoch. I recommend
not turning this to true when the number of epochs
is larger than 500.

'''
model = models.Sequential()
model.add(layers.Dense(2, activation='sigmoid', input_shape=(data.shape[1],)))
#model.add(layers.Dense(2, activation='sigmoid'))
model.add(layers.Dense(1, activation='linear'))

model.compile(optimizer=SGD(lr=0.15),
              loss='mse',
              metrics=['accuracy'])

history = model.fit(data, label,
                    epochs=1000,
                    batch_size=x.shape[0],
                    verbose=False)
#=======================================================#





# This is the networks predictions in a (9, 1)
# array which will be used to plot on the graph
p = model.predict(data)





# ========= Graph Preferences ========= #
plt.plot(x, function_to_approximate, label='Original Function')
plt.plot(x, p, label='Neural Network Prediction', color='red')
plt.legend()
plt.title('Network Approximation vs Function')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.axis('equal')
# set the view of the function by changing the x-axis
# and y-axis values.
plt.ylim((-5,20))
plt.xlim((-10,10))
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
# ===================================== #
