#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# Read data from file
data = np.loadtxt('Lambda_k_band_spin0_bands0_0_old.txt')

# Extract x, y, and z values
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]

# Reshape the data for contour plot
n = int(np.sqrt(len(x)))
X = x.reshape((n, n))
Y = y.reshape((n, n))
Z = z.reshape((n, n))

# Create a 2D plot
fig, ax = plt.subplots()

# Generate the PM3D plot
contour = ax.contourf(X, Y, Z, cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('PM3D Plot (Z-axis Perspective)')

# Add a colorbar
cbar = fig.colorbar(contour)

# Show the plot
plt.show()
