from matplotlib import pyplot as plt
from scipy.sparse.linalg import cg
import cv2
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(description='Color-to-gray conversion')
parser.add_argument('--input', type=str, help='input image path', required=True)
parser.add_argument('--output', type=str, help='output image path', required=True)
parser.add_argument('--mu', type=int, help='neighborhood pixel size', required=True)
parser.add_argument('--npi', type=int, default=1, help="numerator of theta: npi * pi")
parser.add_argument('--dpi', type=int, default=4, help="denominator of theta")
parser.add_argument('--alpha', '-a', type=int, default=20, 
                    help='user parameter alpha')

args = parser.parse_args()

input_img = args.input
output_img = args.output
npi = args.npi
dpi = args.dpi
alpha = args.alpha
theta = npi * math.pi / dpi
u = args.mu 
debug = True

print("start processing image", input_img)

img = cv2.imread(input_img)

img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

height, width, channals = img.shape

pixels = img_lab.astype(int)
# pixels = img_lab

cos_theta, sin_theta = math.cos(theta), math.sin(theta)

ps = pixels.reshape((height * width, 3))
l, a, b = map(list, zip(*ps))

l = list(map(lambda x: x * 100 / 255, l))
a = list(map(lambda x: x - 128, a))
b = list(map(lambda x: x - 128, b))

l_avg = sum(l) / len(l)

pixels = np.array(list(zip(l,a,b))).reshape((height,width,3))

def delta(i, j, alpha, theta):
    da, db = i[1] - j[1], i[2] - j[2]
    dl = i[0] - j[0]
    dist_c = math.sqrt(da ** 2 + db ** 2)
    crunch_dist_c = alpha * math.tanh(dist_c / alpha)
    if abs(dl) > crunch_dist_c:
        return dl
    if da * cos_theta + db * sin_theta >= 0:
        return crunch_dist_c
    return -crunch_dist_c

# Calculate how many pixels belong to each pixel's neighborhood,
# which will be used in constructing matrix A
nneighb = [[0 for _ in range(0, width)] for _ in range(0, height)]
for i in range(0, height):
    for j in range(0, width):
        neighb_top = i - u if i - u >= 0 else 0
        neighb_bot = i + u if i + u <= height - 1 else height - 1
        neighb_left = j - u if j - u >= 0 else 0
        neighb_right = j + u if j + u <= width - 1 else width - 1
        for ni in range(neighb_left, neighb_right+1):
            for nj in range(neighb_top, neighb_bot+1):
                if i * width + j != ni * width + nj:
                    nneighb[i][j] += 1

# Calculate target difference, where deltas[i][j] stores delta_ij
deltas = [[0 for _ in range(0, height * width)] for _ in range(0, height * width)]
for i in range(0, height):
    for j in range(0, width):
        neighb_top = i - u if i - u >= 0 else 0
        neighb_bot = i + u if i + u <= height - 1 else height - 1
        neighb_left = j - u if j - u >= 0 else 0
        neighb_right = j + u if j + u <= width - 1 else width - 1
        for ni in range(neighb_top, neighb_bot+1):
            for nj in range(neighb_left, neighb_right+1):
                deltas[i * width + j][ni * width + nj] = delta(pixels[i][j], pixels[ni][nj], alpha, theta)

# Construct matrix A in the linear system to be solved, where
# A_ij = 2N if i = j, where N is the number of pixels in i's neighborhood
# A_ij = -2 if j in N(i)
# A_ij = 0  otherwise 
diag = []
for row in nneighb:
    for col in row:
        diag.append(2 * col)
A = np.diag(diag)
for i in range(0, height):
    for j in range(0, width):
        neighb_top = i - u if i - u >= 0 else 0
        neighb_bot = i + u if i + u <= height - 1 else height - 1
        neighb_left = j - u if j - u >= 0 else 0
        neighb_right = j + u if j + u <= width - 1 else width - 1
        for ni in range(neighb_top, neighb_bot+1):
            for nj in range(neighb_left, neighb_right+1):
                if i * width + j != ni * width + nj:
                    A[i * width + j][ni * width + nj] = -2
if debug:
    print("********* Finished construct matrix A in linear system *********")
    print(A)

# Consturct vector b in the linear system to be solve, where
# b_i = sum_{j in N(i)} (delta_ij - delta_ji)
B = np.zeros((height * width,))
for i in range(0, height):
    for j in range(0, width):
        neighb_top = i - u if i - u >= 0 else 0
        neighb_bot = i + u if i + u <= height - 1 else height - 1
        neighb_left = j - u if j - u >= 0 else 0
        neighb_right = j + u if j + u <= width - 1 else width - 1
        for ni in range(neighb_top, neighb_bot+1):
            for nj in range(neighb_left, neighb_right+1):
                B[i * width + j] += deltas[i * width + j][ni * width + nj] - deltas[ni * width + nj][i * width + j]


g_flat = np.asarray([[pixels[row][col][0] for col in range(0,width)] for row in range(0,height)]).flatten()


plt.imshow(np.reshape(g_flat, (height, width)), cmap='gray', vmin=0, vmax=255)

res, info = cg(A, B, x0=g_flat)

res = res + (l_avg - res.mean())

res = list(map(lambda x: x * 255 / 100, res))

out = np.reshape(res,(height,width))

# out = out + (l_avg - out.mean())

if debug:
    print("********* Finished solving linear system *********")
    print(out)
    print("status", info)

plt.imshow(out, cmap='gray', vmin=0, vmax=255)
cv2.imwrite(output_img, out)

print("finished processing image", input_img, 
      "output image to", output_img)