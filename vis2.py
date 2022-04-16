import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
coords1 = []
with open('run1/p1.csv') as f:
    for line in f.readlines():
        x, y, z = map(float, re.sub(r'[\[ \];]','', line).split(","))
        coords1.append((x, y, z))
    coords1 = np.stack(coords1).T
coords2 = []
with open('run1/p2.csv') as f:
    for line in f.readlines():
        x, y, z = map(float, re.sub(r'[\[ \];]','', line).split(","))
        coords2.append((x, y, z))
    coords2 = np.stack(coords2).T
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(coords1[0], coords1[1], coords1[2])
ax.scatter(coords2[0], coords2[1], coords2[2])
plt.show()
