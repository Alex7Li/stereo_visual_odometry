import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
import sys
from scipy.spatial.transform import Rotation as R


def main(foldername):
    xs, ys, zs, gtxs, gtys = [],[],[],[],[]
    with open(foldername + '/result.csv') as f:
        f.readline()
        x,y,z,x_0,y_0 = map(float, f.readline().split(','))
        xs.append(x)
        ys.append(y)
        zs.append(z)
        gtxs.append(0)
        gtys.append(0)
        for line in f.readlines():
            x,y,z,gtx,gty = map(float, line.split(','))
            xs.append(x)
            ys.append(y)
            zs.append(z)
            gtxs.append(gtx - x_0)
            gtys.append(gty - y_0)
    for i in range(len(xs)):
        # gt: (x,y,z) = (backward, side, up) | robot: (x,y,z) = (side, up, forward)
        xs[i], ys[i], zs[i] =  -zs[i], xs[i], ys[i]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, color=colors)
    ax.scatter(gtxs, gtys, [0 for _ in gtxs], color=colors)
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
        f"End={xs[-1]:.9f}, {ys[-1]:.9f}, {zs[-1]:.9f}")
    plt.show()
def printRed(text):
        print(f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m")
    
if __name__ == '__main__':
    if len(sys.argv) < 2:
        printRed("No folder specified")
        foldername = 'cfe_cameras';
    else:
        foldername = sys.argv[1];
    main(foldername)