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
        z,y,x,gtx,gty = map(float, line.split(','))
        
        xs.append(-x)
        ys.append(y)
        zs.append(z)
        gtxs.append(gtx - x_0)
        gtys.append(gty - y_0)
# for i in range(128):
#     dd = np.linalg.norm([xs[i + 1] - xs[i],ys[i + 1]-ys[i],zs[i + 1]-zs[i]])
#     d2 = np.linalg.norm([gtxs[i + 1] - gtxs[i],gtys[i + 1]-gtys[i]])
#     print(f'dd: {dd} d2: {d2}')

colors = cm.rainbow(np.linspace(0, 1, len(ys)))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(xs, ys, zs, color=colors)
ax.scatter(gtxs, gtys, [0 for _ in gtxs], color='b')
ax.axes.set_xlim3d(left=-4.5, right=0) 
ax.axes.set_ylim3d(bottom=-2, top=2)
ax.axes.set_zlim3d(bottom=-2, top=2)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
distance_from_goal = np.linalg.norm([xs[-1]-gtxs[-1],ys[-1] - gtys[-1],zs[-1]])
goal_distance = np.sqrt(gtxs[-1]**2 + gtys[-1]**2)
print(f"Absolute Error={distance_from_goal:.4f}, Goal distance={goal_distance:.4f}, "
      f"Relative Error={np.abs(distance_from_goal)/goal_distance:.4f}"
      f"Total D={np.sqrt(np.abs(xs[-1]**2 + ys[-1]**2 + zs[-1]**2))}"
      f"End={xs[-1]:.2f}, {ys[-1]:.2f}, {zs[-1]:.2f}")
plt.show()

