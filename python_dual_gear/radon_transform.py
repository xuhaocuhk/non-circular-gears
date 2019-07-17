import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.data import camera
from skimage.transform import radon, rescale

image = imread('horse.png', True)
image = rescale(image, scale=1.0, mode='reflect', multichannel=False)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4.5))

ax1.set_title("Original")
ax1.imshow(image, cmap=plt.cm.Greys_r)

theta = np.linspace(0., 180., max(image.shape), endpoint=False)
sinogram = radon(image, theta=theta, circle=True)
ax2.set_title("Radon transform\n(Sinogram)")
ax2.set_xlabel("Projection angle (deg)")
ax2.set_ylabel("Projection position (pixels)")
ax2.imshow(sinogram, cmap=plt.cm.Greys_r,
           extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
squaresino = [sum(np.square(column)) for column in sinogram]
ax3.set_title('Rf(theta)')
ax3.plot(squaresino)
fig.tight_layout()
plt.show()
