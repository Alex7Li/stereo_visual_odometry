import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
xs, ys, gtxs, gtys = [],[],[],[]
with open('run1/result.csv') as f:
    f.readline()
    _, _, x_0, y_0 = map(float, f.readline().split(','))
    for line in f.readlines():
        x,y,gtx,gty = map(float, line.split(','))
        xs.append(x)
        ys.append(y)
        gtxs.append(gtx - x_0)
        gtys.append(gty - y_0)
def make_R(theta):
    # return np.eye(2)
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
mp = len(xs)//2
theta1 = math.atan2(ys[mp], xs[mp])
theta2 = math.atan2(gtys[mp], gtxs[mp])
print(f"Adjusting angle by {theta1-theta2}")
R1 = make_R(-theta1)
R2 = make_R(-theta2)
for i in range(len(xs)):
    xs[i], ys[i] = R1 @ np.array([xs[i], ys[i]])
    gtxs[i], gtys[i] = R2 @ np.array([gtxs[i], gtys[i]])
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
plt.scatter(xs, ys, color=colors)
plt.scatter(gtxs, gtys, color='b')
plt.xlim(0,4)
plt.ylim(-2,2)
plt.show()

