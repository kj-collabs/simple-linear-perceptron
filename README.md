# Simple Linear Perceptron Classifier with a Graphical User Interface

This application allows users to visualise the <a href="https://en.wikipedia.org/wiki/Perceptron">Perceptron Algorithm</a> of a two-class classification problem. The linear boundary shifts and adjusts until either the number of iterations has reached a maximum, or the two classes have been classified correctly by the boundary.

### The Graph
This section displays the points, where *blue dots* represent *class 1* and *red crosses* represent *class 2*.

### Update Axes Limit
Allows the user to set a *lower-bound* and *upper bound* on the x-axis and y-axis. Pressing *'Update Axes'* updates the graph axes, and if points are already plotted, they will scale accordingly.

### Add Points to Dataset
##### Entering Points Manually:
The *x* and *y* coordinate of each point can be specified manually and a *class* for that point can be chosen before adding it to the graph. The user can add as many points as they wish.
##### Generate Random Dataset of Points 
This feature allows the user to generate random points which is *linearly seperable*, therefore it guarantees the Perceptron to divide the two classes.

### Run Perceptron
The *initial weights* must be specified, w1, w2, and w0 (bias). This is the initial boundary/line through the dataset. An appropriate *learning rate* must be set and an *iteration limit*. The iteration limit becomes important for classifications which are not linearly seperable as it will cause the algorithm to run infinitely. The user can choose the *speed* at which the linear boundary moves, using a slider, before running the algorithm.
Click *'Run Perceptron'* to watch the linear boundary move to seperate the two class classification.


