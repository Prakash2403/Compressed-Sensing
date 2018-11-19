import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def haar_matrix(nm, normalized=False):
    """
    Function to calculate haar matrix of dimension nm x nm.
    :param nm: dimension of the haar matrix.
    :param normalized: If true, normalize the haar matrix.
    :return: matrix of dimension nm x nm.
    """
    nm = 2 ** np.ceil(np.log2(nm))
    if nm > 2:
        hm = haar_matrix(nm / 2)
    else:
        return np.array([[1, 1], [1, -1]])

    h_n = np.kron(hm, [1, 1])
    if normalized:
        h_i = np.sqrt(nm / 2) * np.kron(np.eye(len(hm)), [1, -1])
    else:
        h_i = np.kron(np.eye(len(hm)), [1, -1])
    hm = np.vstack((h_n, h_i))
    return hm


img = cv2.imread('images/black-white-animal.jpg', cv2.IMREAD_GRAYSCALE)
resized_img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)   # Resizing the image for better performance.
a, b = resized_img.shape
plt.imshow(resized_img.reshape((a, b)), cmap=plt.cm.gray)   # Plotting reconstructed image after L1 optimization.
plt.show()
N = a * b
n = 64 * 32
resized_img = np.reshape(resized_img, (N, 1))   # Vectorizing the image.
h = haar_matrix(N)  # Creating N x N haar matrix.
h_inv = np.linalg.inv(h)
c = np.matmul(h_inv, resized_img)   # Calculating coefficients. Next few lines will check for signal compressibilty.
temp_c = np.abs(c)
temp_c.sort(axis=0)
temp_c = np.flip(temp_c)
plt.plot(temp_c)  # Visual check for signal compressibility.
plt.show()
t = temp_c[len(c)//5]   # Taking top 20% DCT coefficients.
c[abs(c) < t] = 0
reconstructed_img = np.matmul(h, c)
plt.imshow(reconstructed_img.reshape((a, b)), cmap=plt.cm.gray)  # Plotting reconstructed image after taking top 20%
plt.show()                                                       # DCT coefficients.
A = np.random.normal(size=(n, N))   # Creating measurement matrix, dimensions n x N
y = np.matmul(A, resized_img)  # Calcutlating measurements.
x_bar = cp.Variable((N, 1))     # Formulating the problem, next few lines minimize |x_bar|_l1, s.t. ||y - Ahx_bar|| < s.
objective = cp.Minimize(cp.norm(x_bar, p=1))    # Where s is standard deviation of noise.
constraints = [y - np.matmul(A, h) * x_bar <= 0, y - np.matmul(A, h) * x_bar >= 0]  # Defining the constraints.
prob = cp.Problem(objective, constraints)
result = prob.solve()
theta = x_bar.value
rec_img = np.matmul(h, theta)   # Reconstructing the image using vector obtained after l1 minimization.
rec_img = rec_img - np.min(rec_img)  # Optional, not needed.
cv2.imwrite("animal.png", rec_img.reshape(a, b))
plt.imshow(rec_img.reshape((a, b)), cmap=plt.cm.gray)   # Plotting reconstructed image after L1 optimization.
plt.show()
