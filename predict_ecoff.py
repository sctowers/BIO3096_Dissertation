import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm

# plot a histogram
x_cols = ['0.002', '0.004', '0.008',
          '0.016', '0.03', '0.06', '0.125',
          '0.25', '0.5', '1', '2', '4', '8',
          '16', '32', '64', '128', '256', '512']
y_col = '(T)ECOFF'

df = pd.read_csv('reduced_ecoff_values.csv')
full_x = df[x_cols].to_numpy()
full_y = df[y_col].to_numpy()

f'{full_x.shape=}, {full_y.shape=}'

plt.hist(full_y, bins=30)

# true vs predicted ECOFF
train_x, test_x, train_y, test_y = train_test_split(full_x, full_y, train_size=0.80)
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

neural_network = MLPRegressor(hidden_layer_sizes=(50, 50, 50), )
neural_network = neural_network.fit(train_x, train_y)

training_error = neural_network.loss_curve_
train_prediction = neural_network.predict(train_x)
test_prediction = neural_network.predict(test_x)
train_loss = mean_squared_error(train_y, train_prediction)
test_loss = mean_squared_error(test_y, test_prediction)

f'{train_loss=}, {test_loss=}'
# plot the training error
plt.plot(np.arange(len(training_error)), training_error)

# scatter plot
fig, ax = plt.subplots()

ax.scatter(train_prediction, train_y, label="Train samples", c='blue')
ax.scatter(test_prediction, test_y, label="Test samples", c='green')
ax.set_xlabel("Predicted value")
ax.set_ylabel("True value")
ax.legend()

# normalised ECOFF
# print(full_x.shape, np.sum(full_x, axis=1).shape)
full_x_norm = (full_x.T / np.sum(full_x, axis=1)).T
# print(full_x_norm.shape)
# print(full_x[0])
# print(full_x_norm[0])
# print(np.sum(full_x_norm[0]))

train_x, test_x, train_y, test_y = train_test_split(full_x_norm, full_y, train_size=0.80)
print(f'{train_x.shape=}, {test_x.shape=}, {train_y.shape=}, {test_y.shape=}')

neural_network = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
neural_network = neural_network.fit(train_x, train_y)

training_error = neural_network.loss_curve_
train_prediction = neural_network.predict(train_x)
test_prediction = neural_network.predict(test_x)
train_loss = mean_squared_error(train_y, train_prediction)
test_loss = mean_squared_error(test_y, test_prediction)

f'{train_loss=}, {test_loss=}'

plt.plot(np.arange(len(training_error)), training_error)

fig, ax = plt.subplots()

ax.scatter(train_prediction, train_y, label="Train samples", c='blue')
ax.scatter(test_prediction, test_y, label="Test samples", c='green')
ax.set_xlabel("Predicted value")
ax.set_ylabel("True value")
ax.legend()

# compare ECOFF training vs testing loss for 100 ANNs
n = 10
training_losses = []
testing_losses = []

for _ in tqdm(range(n)):
    train_x, test_x, train_y, test_y = train_test_split(full_x_norm, full_y, train_size=0.80)

    neural_network = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=1000)
    neural_network = neural_network.fit(train_x, train_y)

    train_loss = mean_squared_error(train_y, neural_network.predict(train_x))
    test_loss = mean_squared_error(test_y, neural_network.predict(test_x))
    print(f'{train_loss=}, {test_loss=}')

    training_losses.append(train_loss)
    testing_losses.append(test_loss)

# Create a Figure, which doesn't have to be square.
fig = plt.figure(layout='constrained')
# Create the main axes, leaving 25% of the figure space at the top and on the
# right to position marginals.
ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
# The main axes' aspect can be fixed.
ax.set(aspect=1)
# Create marginal axes, which have 25% of the size of the main axes.  Note that
# the inset axes are positioned *outside* (on the right and the top) of the
# main axes, by specifying axes coordinates greater than 1.  Axes coordinates
# less than 0 would likewise specify positions on the left and the bottom of
# the main axes.
ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
# Draw the scatter plot and marginals.
# no labels
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)

# the scatter plot:
ax.scatter(training_losses, testing_losses)

# now determine nice limits by hand:
# binwidth = 0.25
# xymax = max(np.max(np.abs(training_losses)), np.max(np.abs(testing_losses)))
# lim = (int(xymax/binwidth) + 1) * binwidth

# bins = np.arange(-lim, lim + binwidth, binwidth)
ax_histx.hist(training_losses)
ax_histy.hist(testing_losses, orientation='horizontal')
