import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
coords = []
with open('dbg.csv') as f:
    for line in f.readlines():
        x, y, z = map(float, re.sub(r'[\[ \];]','', line).split(","))
        coords.append((x, y, z))
    coords = np.stack(coords).T
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(coords[0], coords[1], coords[2])
plt.show()
