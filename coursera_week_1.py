import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from scipy import signal
from numpy import pi, cos, sin


def intensity_level(img, new_level):
    if np.log2(new_level) % 1 != 0:
        raise ValueError
    if len(img.shape) > 2:
        img_trans = np.copy(img)
        length, width, _ = img.shape
        for k in range(3):
            img_trans[:, :, k] = (np.floor(img[:, :, k] * (new_level / 256)) / (new_level / 256)).reshape(length, width)
    else:
        length, width = img.shape
        img_trans = (np.floor(img.ravel() * (new_level / 256)) / (new_level / 256)).reshape(length, width)
    return img_trans


def kernel_blur(img, kernel_dim):
    img_trans = np.copy(img)
    nhood = np.ones([kernel_dim, kernel_dim]) / kernel_dim ** 2
    if len(img.shape) > 2:
        for k in range(3):
            img_trans[:, :, k] = signal.convolve2d(img[:, :, k], nhood, 'same')
    else:
        img_trans = signal.convolve2d(img, nhood, 'same')
    return img_trans


def downsample(img, kernel_dim):
    img_trans = kernel_blur(img, kernel_dim)
    if len(img.shape) > 2:
        length, width, _ = np.array(img.shape) + 1 - kernel_dim
        img_trans = img_trans[0:length:kernel_dim, 0:width:kernel_dim, :]
    else:
        length, width = np.array(img.shape) + 1 - kernel_dim
        img_trans = img_trans[0:length:kernel_dim, 0:width:kernel_dim]
    return img_trans


img = io.imread(r'C:\Users\hugoc\Pictures\upsidedownplane.jpg')
img_bw = 0.2125 * img[:, :, 0] + 0.7154 * img[:, :, 1] + 0.0721 * img[:, :, 2]
# Changing intensity levels
#img_trans = intensity_level(img, 4)
#fig, ax = plt.subplots()
#ax.imshow(img_trans)
#plt.show()

# Blurring through 2D convolution
#fig2, ax2 = plt.subplots()
#img_trans2 = kernel_blur(img, 10)
#ax2.imshow(img_trans2)
#plt.show()

# Rotation
angle = pi/4
fig3, ax3 = plt.subplots()
length, width = img_bw.shape
x, y = np.meshgrid(np.arange(0, width, 1), np.arange(length-1, -1, -1))
X = x*cos(angle) - y*sin(angle)
Y = x*sin(angle) + y*cos(angle)
cart_prod = np.dstack((x.ravel(), y.ravel())).reshape((width, length, 2)).reshape(-1, 2)
CART_PROD = np.dstack((X.ravel(), Y.ravel())).reshape((width, length, 2)).reshape(-1, 2)
x_max = round(CART_PROD[-1, 0] - CART_PROD[0, 0])
y_max = round(CART_PROD[width-1, 1] - CART_PROD[-width, 1])
# (0, pi/2)|(pi/2, pi)|(pi, 3pi/2)|(3pi/2, 2pi)
# UpLe turns into Le, Lo, Ri, Up
# LoLe turns into Lo, Ri, Up, Le
# LoRi turns into Ri, Up, Le, Lo
# UpRi turns into Up, Le, Lo, Ri
# Tri1=Tri3: x = width*cos(angle), y = width*sin(angle)
# Tri2=Tri4: x = length*sin(angle), y = length*cos(angle)
x_new, y_new = np.meshgrid(np.arange(0, x_max, 1), np.arange(y_max-1, -1, -1))
cart_prod_new = np.dstack((x_new.ravel(), y_new.ravel())).reshape((x_max, y_max, 2)).reshape(-1, 2)
plt.scatter(x=cart_prod_new[:, 0], y=cart_prod_new[:, 1], marker='.')
plt.scatter(x=CART_PROD[:, 0]+length*sin(angle), y=CART_PROD[:, 1], marker='.')
plt.axis('equal')
plt.show()
print(cart_prod)
print(img_bw.shape)


# Downsampling
#fig4, ax4 = plt.subplots()
#img_trans4 = downsample(img, 4)
#ax4.imshow(img_trans4)
#plt.show()

