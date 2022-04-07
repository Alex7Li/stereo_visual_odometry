import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
xs, ys, zs, gtxs, gtys = [],[],[],[],[]
foldername = 'run1';
# foldername = 'rand_feats';
with open(foldername + '/result.csv') as f:
    f.readline()
    _, _, _, x_0, y_0 = map(float, f.readline().split(','))
    for line in f.readlines():
        x,y,z,gtx,gty = map(float, line.split(','))
        xs.append(x)
        ys.append(y)
        zs.append(z)
        gtxs.append(gtx - x_0)
        gtys.append(gty - y_0)
# for i in range(50):
#     dd = np.linalg.norm([xs[i + 1] - xs[i],ys[i + 1]-ys[i],zs[i + 1]-zs[i]])
#     d2 = np.linalg.norm([gtxs[i + 1] - gtxs[i],gtys[i + 1]-gtys[i]])
#     print(f'dd: {dd} d2: {d2}')

def make_R(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta),0],
        [np.sin(theta), np.cos(theta),0],
        [0,0,1],
    ])

mp = len(xs)//4
theta1 = math.atan2(ys[mp], xs[mp])
theta2 = math.atan2(gtys[mp], gtxs[mp])
print(f"Adjusting angle by {theta1-theta2}")
R1 = make_R(-theta1)
R2 = make_R(-theta2)
for i in range(len(xs)):
    xs[i], ys[i], zs[i] = R1 @ np.array([xs[i], ys[i], zs[i]])
    gtxs[i], gtys[i], _ = R2 @ np.array([gtxs[i], gtys[i], 0])
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, color=colors)
ax.scatter(gtxs, gtys, [0 for _ in gtxs], color='b')
# ax.xlim(0, 4.5)
# ax.ylim(-2, 2)
# ax.zlim(-2, 2)
ax.axes.set_xlim3d(left=0, right=4.5) 
ax.axes.set_ylim3d(bottom=-2, top=2) 
ax.axes.set_zlim3d(bottom=-2, top=2) 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

