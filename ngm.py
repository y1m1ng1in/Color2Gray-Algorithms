import cv2
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(description='Color-to-gray conversion')
parser.add_argument('--input', type=str, help='input image path', required=True)
parser.add_argument('--output', type=str, help='output image path', required=True)
parser.add_argument('--dof', type=int, default=4, 
                    help='degree of trigonometric polynomial used')
parser.add_argument('--alpha', '-a', type=float, default=1, 
                    help="user parameter alpha")
parser.add_argument('--lamb', '-l', type=int, default=1, 
                    help="the value times number of pixels for parameter lambda")

args = parser.parse_args()

input_img = args.input
output_img = args.output
n = args.dof
alpha = args.alpha
lamb_times = args.lamb
debug = True

print("start processing image", input_img)

img = cv2.imread(input_img)

height, width, channels = img.shape

def bgr2lab(img):
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    ps = img_lab
    ps = img_lab.astype(float)
    L, A, B = ps[:,:,0], ps[:,:,1], ps[:,:,2]
    L = L * 100 / 255
    A = A - 128
    B = B - 128
    return L, A, B
    
def bgr2luv(img):
    img_luv = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)
    ps=  img_luv
    ps = img_luv.astype(float)
    L, U, V= ps[:,:,0], ps[:,:,1], ps[:,:,2]
    U = np.full((height,width),255) / (354 * U) 
    V = np.full((height,width),255) / (262 * V) 
    return L, U, V

def lab2lch(A,B):
    C = np.sqrt(A ** 2 + B ** 2)
    H = np.arctan2(B, A)
    return C, H

L, A, B = bgr2lab(img)
_, U, V = bgr2luv(img)
C, H = lab2lch(A, B)

if debug:
    print("********* Finished extracting L,A,B,U,V,C,H spaces *********")
    print("L range", np.amax(L),np.amin(L))
    print("A range", np.amax(A),np.amin(A))
    print("B range", np.amax(B),np.amin(B))
    print("U range", np.amax(U),np.amin(U))
    print("V range", np.amax(V),np.amin(V))
    print("C range", np.amax(C),np.amin(C))
    print("H range", np.amax(H),np.amin(H))

img_lab = np.dstack((L, A, B))
img_luv = np.dstack((L, U, V))

def T(theta):
    t = np.zeros(2 * n + 1)
    for i in range(0, n):
        t[i] = math.cos((i + 1) * theta)
    for i in range(n, 2 * n):
        t[i] = math.sin((i - n + 1) * theta)
    t[2 * n] = 1
    return t

def hk_effect(luv):
    l, u, v = luv[0], luv[1], luv[2]
    # La adapting luminance, set by default to 20
    la = 20 
    theta = math.atan(v / u)
    kbr = 0.2717 * (((6.469 + 6.362 * la) ** 0.4495) / (6.469 + la) ** 0.4495)
    suv = 13 * ((u - 0.2009) ** 2 + (v - 0.4610) ** 2) ** 0.5
    q = (
        - 0.01585 
        - 0.03017 * math.cos(theta) 
        - 0.04556 * math.cos(2 * theta) 
        - 0.02677 * math.cos(3 * theta) 
        - 0.00295 * math.cos(4 * theta) 
        + 0.14592 * math.sin(theta) 
        + 0.05084 * math.sin(2 * theta) 
        - 0.01900 * math.sin(3 * theta) 
        - 0.00764 * math.sin(4 * theta))
    # L Gamma_VAC = L + [−0.1340 q(θ) + 0.0872 KBr] suv L
    return l + (-0.1340 * q + 0.0872 * kbr) * suv * l

def cdiff(lab1,lab2,luv1,luv2):
    dl, da, db = lab1[0] - lab2[0], lab1[1] - lab2[1], lab1[2] - lab2[2]
    r = 2.54 * math.sqrt(2)
    ret = math.sqrt(dl ** 2 + (alpha * math.sqrt(da ** 2 + db ** 2) / r) ** 2)
    dlhk = hk_effect(luv1) - hk_effect(luv2)
    if dlhk != 0:
        return ret * np.sign(dlhk)
    if dl != 0:
        return ret * np.sign(dl)
    return ret * np.sign(dl ** 3 + da ** 3 + db ** 3)

Lx, Ly = np.gradient(L)

Gx, Gy = np.zeros((height,width)), np.zeros((height,width))

for i in range(1, height-1):
    for j in range(1, width-1): 
        Gy[i][j] = cdiff(img_lab[i][j + 1], img_lab[i][j - 1],
                         img_luv[i][j + 1], img_luv[i][j - 1]) 
        Gx[i][j] = cdiff(img_lab[i + 1][j], img_lab[i - 1][j],
                         img_luv[i + 1][j], img_luv[i - 1][j])

p, q = Gx - Lx, Gy - Ly

Ct = np.zeros((height, width, 2 * n + 1))
for i in range(0, height):
    for j in range(0, width):
        Ct[i][j] = C[i][j] * T(H[i][j])

u, v, _ = np.gradient(Ct)
    
M, b = np.zeros((2 * n + 1, 2 * n + 1)), np.zeros(2 * n + 1)

for i in range(0, height):
    for j in range(0, width):
        uij = np.array(u[i][j])[np.newaxis]
        vij = np.array(v[i][j])[np.newaxis]
        M = M + uij.T @ uij + vij.T @ vij
        b = b + p[i][j] * u[i][j] + q[i][j] * v[i][j]

lamb = height * width * lamb_times
X = M + lamb * np.identity(2 * n + 1)
X = np.linalg.lstsq(X, b, rcond=None)
X = (np.array(X[0])[np.newaxis]).T

if debug:
    print("********* Finished solving linear system *********")
    print(X)

res_img = np.zeros((height, width))

for i in range(0, height):
    for j in range(0, width):
        f = T(H[i][j]) @ X
        res_img[i][j] = L[i][j] + (C[i][j] * f)

min_val = np.amin(res_img)
max_val = np.amax(res_img)
if debug:
    print("********* Finished global mapping *********")
    print(max_val, min_val)

for i in range(0, height):
    for j in range(0, width):
        res_img[i][j] = ((res_img[i][j] - min_val) / (max_val - min_val)) * 255

cv2.imwrite(output_img, res_img)

print("finished processing image", input_img, 
      "output image to", output_img)