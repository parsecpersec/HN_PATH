from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time
im = Image.open('C:/users/Me/Desktop/img.jpg')
g_im = im.convert('L')

# Gaussian filter
def gauss(sigma=1.0, size=5):
    mask = np.empty([size, size])
    if size % 2 == 1:
        k = (size-1)/2
        total = 0
        for i in range(size):
            for j in range(size):
                mask[i][j] = pow(np.e, -((i-k)**2+(j-k)**2)/(2*sigma*sigma))
                total += mask[i][j]
        for i in range(size):
            for j in range(size):
                mask[i][j] /= total
    return mask

# convolution
def conv(img_array, kernel):
    s = len(kernel)
    k = int((s-1)/2)
    h = img_array.shape[0]
    w = img_array.shape[1]
    filtered = np.empty([h, w])
    sub_im = np.empty([s, s])
    for i in range(h):
        for j in range(w):
            for m in range(i-k, i+k+1):
                for n in range(j-k, j+k+1):
                    tm = m
                    tn = n
                    if m < 0:
                        m = 0
                    elif m >= h:
                        m = h-1
                    if n < 0:
                        n = 0
                    elif n >= w:
                        n = w-1
                    sub_im[tm-i+k][tn-j+k] = img_array[m][n]
            filtered[i][j] = sum(sum(sub_im * kernel))
    return filtered

print(time.strftime('%H:%M:%S'))
f_im = conv(np.asarray(g_im), gauss(1.4))
print(time.strftime('%H:%M:%S'))
plt.subplot(1, 2, 1), plt.imshow(g_im, cmap='gray')
plt.subplot(1, 2, 2), plt.imshow(Image.fromarray(f_im), cmap='gray')
plt.show()
