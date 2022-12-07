import numpy as np
import matplotlib.pyplot as plt
import time


def draw(x1, x2):
    ln = ax[0].plot(x1, x2)
    plt.pause(0.00001)
    ln[0].remove()

def draw_error (error_number, t):
    #ln = ax[0].plot(x1, x2)
   # plt.pause(0.00001)
   # ln[0].remove()
    print (t)
    ax[1].scatter(t, error_number)

def sigmoid(score):
    return 1 / (1 + np.exp(-score))


def calculate_error(line_parameters, points, y):
    m = points.shape[0]
    p = sigmoid(all_points * line_parameters)
    cross_entropy = -(1 / m) * (np.log(p).transpose() * y + np.log(1 - p).transpose() * (1 - y))
    return cross_entropy


def gradient_descent(line_parameters, points, y, alpha):
    m = points.shape[0]
    error_list = [1]
    t=0
    for i in range(100):
        p = sigmoid(all_points * line_parameters)
        gradient = (points.transpose() * (p - y)) * (alpha / m)
        line_parameters = line_parameters - gradient
        w1 = line_parameters.item(0)
        w2 = line_parameters.item(1)
        b = line_parameters.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = - b / w2 + x1 * (-w1 / w2)
        draw(x1, x2)
        print(calculate_error(line_parameters, points, y))
        error = calculate_error(line_parameters, points, y)
        error_number = error[0,0]
        t += 1
        error_list.append(error_number)
        draw_error(error_number, t)
    ln = ax[0].plot(x1, x2)
    return error_list


n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 3, n_pts), bias]).transpose()
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 3, n_pts), bias]).transpose()
all_points = np.vstack((top_region, bottom_region))
line_parameters = np.matrix([np.zeros(3)]).transpose()
# x1 = np.array([bottom_region[:,0].min(), top_region[:,0].max()])
# x2 = - b / w2 + x1 * (-w1/w2)
y = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)

_, ax = plt.subplots(1,2)
ax[0].scatter(top_region[:, 0], top_region[:, 1], color='r')
ax[0].scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
ax[0].title.set_text("Linear Regression")
ax[1].title.set_text("Error")
error_record = gradient_descent(line_parameters, all_points, y, 0.06)
#plt.show()

#_, ax2 = plt.subplots(figsize=(4, 4))
#t = range(101)
#print(error_record)
#print(t)
#ax[1].plot(t, error_record)
plt.show()
#print(calculate_error(line_parameters, all_points, y))
