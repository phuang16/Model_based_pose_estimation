import numpy as np
import math
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

# Read in the image
I = mpimg.imread('img1_rect.tif')
# I = mpimg.imread('img2_rect.tif')

plt.figure(0)
plt.imshow(I)

########## Estimate pose ########################
# 6 feature points represented in the model's coordinate (unit: inches)
P_M = np.array([[0, 0, 2, 0, 0, 2],
        [10, 2, 0, 10, 2, 0],
        [6, 6, 6, 2, 2, 2],
        [1, 1, 1, 1, 1, 1]])

# Define camera parameters(unit: pixels)
f = 715  # focal length
cx = 354  # center points
cy = 245

# Intrinsic camera parameter matrix
K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
# print(K)

# Observed image points
# For "img1_rect.tif"
p = np.array([[183, 350, 454, 176, 339, 444],
              [147, 133, 144, 258, 275, 286],
              [1, 1, 1, 1, 1, 1]])
# # For "img2_rect.tif"
# p = np.array([[107, 297, 398, 128, 314, 413],
#     [212, 202, 191, 324, 339, 326],
#     [1, 1, 1, 1, 1, 1]])

N = p.shape[1]
p_Transpose = np.transpose(p[0:2, ])
y0 = p_Transpose.reshape(N * 2, 1)

# Initial guess of the pose: [ax ay az tx ty tz]
x = np.array([1.5, -1.0, 0.0, 0, 0, 30])

# Project 3D points (in model) onto 2D image
def fProject(x, P_M, K):
    # Get pose parameters
    ax = x[0]; ay = x[1]; az = x[2]
    tx = x[3]; ty = x[4]; tz = x[5]

    # Rotation matrix, model to camera
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    # Extrinsic camera matrix
    Mext = np.append(R, np.array([[tx], [ty], [tz]]), axis=1)

    # Project points
    ph = np.dot(K, np.dot(Mext, P_M))

    # Divide through 3rd element of each column
    ph[0, :] = ph[0, :] / ph[2, :];
    ph[1, :] = ph[1, :] / ph[2, :];
    ph2 = np.transpose(np.array([ph[0,], ph[1,]]))  # Get rid of 3rd row
    p = ph2.reshape(N * 2, 1)  # reshape into 2Nx1 vector
    return (p)

# Get predicted image points by substituting in the current pose
y = fProject(x, P_M, K)

# Plot the predicted image points
plt.figure(0)
plt.imshow(I)
plt.title('initial y')
for i in range(0, len(y), 2):
    rectangle = plt.Rectangle((y[i]-8, y[i+1]-8), 16, 16, fc='r')
    plt.gca().add_patch(rectangle)

plt.figure(0)
plt.imshow(I)

for i in range(0, 10):
    print("\nIteration %d Current pose:\n" % (i), x)
    # Get predicted image points
    y = fProject(x, P_M, K)

    # Plot predicted results at each iteration
    plt.figure(i)
    plt.imshow(I)
    plt.title('Iteration ' + str(i))
    for j in range(0, len(y), 2):
        rectangle = plt.Rectangle((y[j] - 8, y[j + 1] - 8), 16, 16, fc='r')
        plt.gca().add_patch(rectangle)

    # Estimate Jacobian
    e = 1e-5
    temp_dx = np.array(np.zeros(6))
    temp_dx[0] = e
    J_old = ((fProject(x + temp_dx, P_M, K) - y) / e)
    for j in range(1, 6):
        temp_dx = np.array(np.zeros(6))
        temp_dx[j] = e
        J_new = ((fProject(x + temp_dx, P_M, K) - y) / e)
        J = np.hstack([J_old, J_new])
        J_old = J

    dy = y0 - y
    print("Residual error of %f\n" % np.linalg.norm(dy, 2))

    # We have dy = J * dx
    # Can obtain dx by dx = pseudo inverse J * dy
    dx = np.dot(np.linalg.pinv(J), dy)

    # Stop if changes in x parameters are negligible
    if np.absolute(np.linalg.norm(dx, 2) / np.linalg.norm(x, 2)) < 1e-6:
        break
    x = x + np.transpose(dx[:, 0])

# Print final pose
ax = x[0]; ay = x[1]; az = x[2]
tx = x[3]; ty = x[4]; tz = x[5]

# Rotation matrix (from model to camera)
Rx = np.array([[1, 0, 0],
               [0, np.cos(ax), -np.sin(ax)],
               [0, np.sin(ax), np.cos(ax)]])

Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
               [0, 1, 0],
               [-np.sin(ay), 0, np.cos(ay)]])

Rz = np.array([[np.cos(az), -np.sin(az), 0],
               [np.sin(az), np.cos(az), 0],
               [0, 0, 1]])
R = np.dot(Rz, np.dot(Ry, Rx))

# Extrinsic camera parameter matrix
Mext = np.append(R, np.array([[tx], [ty], [tz]]), axis=1)
print("\nFinal pose:", Mext)

####### Estimate uncertainty ########
# Sum of squared error
sse = np.linalg.norm(dy,2)**2

# Estimated image standard deviation
sigmaImg = math.sqrt(sse/(2*N-6))
print("#pts = %d, estimated image error = %f pixels\n" % (N, sigmaImg))

# Covariance of the measured image points
Cp = np.diag((sigmaImg**2) * np.ones(2*N))
L = np.dot(np.linalg.inv((np.dot(np.transpose(J), J))), np.transpose(J))
C = np.dot(L, np.dot(Cp, np.transpose(L)))
# print(C)

# -------- Plot uncertainty --------
# Covariance of translation portion
Cz = C[3:6, 3:6]

# Draw ellipsoid (defined by the surface x' Cinv x = z^2, where Cinv = Cz^-1)
# (For z=3, this is the 97% probability surface)
# First, center the ellipsoid at the location of the model
xc = tx; yc = ty; zc = tz

# Find the rotation matrix R s.t. the ellipsoid aligns with the axes
# Let y = Rx,
# Then y'*D*y = z^2 (where D is a diagonal matrix)
# Or, x'R'*D*Rx = z^2.  x'(R'DR)x = z^2.
# So Cinv = R'DR.  This is just taking the SVD of Cinv.
U, S, V = np.linalg.svd(np.linalg.inv(Cz), full_matrices=True)
R = V

# The length of the ellipsoid axes are len=1/sqrt(Si/z^2)
# (where Si is the ith eigenvalue)
xrad = math.sqrt(9/(S[0]))
yrad = math.sqrt(9/S[1])
zrad = math.sqrt(9/S[2])

# plot the uncertainty ellipsoid
fig = plt.figure()
ax = fig.gca(projection='3d')
plt.title('Uncertainty')

# Set spherical angles
Nsteps = 100;
phi = np.linspace(0, 2*np.pi, Nsteps)
theta = np.linspace(0, np.pi, Nsteps)

# Parameterize the ellipsoid with sphereical coordinates
# theta: polar angle / phi: azumith angle
x = xrad * np.outer(np.sin(theta), np.cos(phi))
y = yrad * np.outer(np.sin(theta), np.sin(phi))
z = zrad * np.outer(np.cos(theta), np.ones_like(phi))
xrow, xcol = np.shape(x)

# Rotate the ellipsoid
for i in range(0, xrow):
    for j in range(0, xcol):
        Y = np.dot(np.transpose(R), np.array([x[i, j], y[i, j], z[i, j]]))
        x[i, j] = Y[0]+xc
        y[i, j] = Y[1]+yc
        z[i, j] = Y[2]+zc

ax.plot_surface(x, y, z)
plt.show()

###### Draw everything in 3D #######
fig = plt.figure(figsize=plt.figaspect(1.2))
ax = fig.gca(projection='3d')

# ----  Draw Uncertainty ellipsoid --------
# ----- (Obtained previously) -----------
ax.plot_surface(x, y, z)

# ----  Draw box (model) --------
# Define line segments on the box
w = 11.5; h = 10.0; d = 17.0

pStart = np.array([[0,0,0,0,0,0,0,w,w,0,w,w],
                  [0,0,0,0,0,d,d,0,0,d,0,d],
                  [0,0,0,h,h,0,0,0,0,h,h,0],
                  [1,1,1,1,1,1,1,1,1,1,1,1]])
pEnd = np.array([[w,0,0,w,0,0,w,w,w,w,w,w],
                  [0,d,0,0,d,d,d,d,0,d,d,d],
                  [0,0,h,h,h,h,0,0,h,h,h,h],
                  [1,1,1,1,1,1,1,1,1,1,1,1]])
H = np.append(Mext, np.array([[0,0,0,1]]), axis =0)

pStart_C = np.dot(H, pStart)
pEnd_C = np.dot(H, pEnd)

# Wireframe model
N_lines = np.shape(pStart)
N_lines = N_lines[1]
for i in range(0, N_lines):
    ax.plot([pStart_C[0,i], pEnd_C[0,i]],
            [pStart_C[1, i],pEnd_C[1, i]],
            zs=[pStart_C[2, i],pEnd_C[2, i]],
           c='m')

# Plot coordinate axes
pAxes = np.dot(H, np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4], [1, 1, 1]]))

ax.plot([H[0, 3], pAxes[0, 0]],
        [H[1, 3], pAxes[1, 0]],
        zs=[H[2, 3], pAxes[2, 0]],
        c='r', linewidth=3)
ax.plot([H[0, 3], pAxes[0, 1]],
        [H[1, 3], pAxes[1, 1]],
        zs=[H[2, 3], pAxes[2, 1]],
        c='g', linewidth=3)
ax.plot([H[0, 3], pAxes[0, 2]],
        [H[1, 3], pAxes[1, 2]],
        zs=[H[2, 3], pAxes[2, 2]],
        c='b', linewidth=3)

# Targets
P_c = np.dot(H, P_M)
ax.scatter(P_c[0,:], P_c[1,:], P_c[2,:], c='k', marker='+')
# plt.show()


# ----  Draw camera at the origin --------
# Define the vertices
Ver = np.array([[-1, -1, -2],
              [ 1, -1, -2],
              [ 1,  1, -2],
              [-1,  1, -2],
              [-1, -1,  2],
              [ 1, -1,  2],
              [ 1,  1,  2],
              [-1,  1,  2]])

# # plot vertices
# ax.scatter3D(Ver[:, 0], Ver[:, 1], Ver[:, 2])

# All 6 sides of the polygon
allsides = [[Ver[0],Ver[1],Ver[2],Ver[3]],
 [Ver[4],Ver[5],Ver[6],Ver[7]],
 [Ver[1],Ver[2],Ver[6],Ver[5]],
 [Ver[4],Ver[7],Ver[3],Ver[0]],
 [Ver[0],Ver[1],Ver[5],Ver[4]],
 [Ver[2],Ver[3],Ver[7],Ver[6]]]

# Plot all sides
ax.add_collection3d(Poly3DCollection(allsides,
 facecolors='c', edgecolors='k', alpha=.75))

ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')

# set the x, y, z display range
ax.set_xlim3d(-20, 20)
ax.set_ylim3d(-20, 20)
ax.set_zlim3d(-5, 40)

ax.view_init(elev = -68, azim=-88)


plt.show()

# ----  Plot a zoom in area for the uncertainty ellipsoid  --------
# ----- (at the model coordinate axes) ------

fig = plt.figure(figsize=plt.figaspect(1.2))
ax = fig.gca(projection='3d')

# Wireframe model
N_lines = np.shape(pStart)
N_lines = N_lines[1]
for i in range(0, N_lines):
    ax.plot([pStart_C[0,i], pEnd_C[0,i]],
            [pStart_C[1, i],pEnd_C[1, i]],
            zs=[pStart_C[2, i],pEnd_C[2, i]],
           c='m')

ax.plot([H[0, 3], pAxes[0, 0]],
        [H[1, 3], pAxes[1, 0]],
        zs=[H[2, 3], pAxes[2, 0]],
        c='r', linewidth=3)
ax.plot([H[0, 3], pAxes[0, 1]],
        [H[1, 3], pAxes[1, 1]],
        zs=[H[2, 3], pAxes[2, 1]],
        c='g', linewidth=3)
ax.plot([H[0, 3], pAxes[0, 2]],
        [H[1, 3], pAxes[1, 2]],
        zs=[H[2, 3], pAxes[2, 2]],
        c='b', linewidth=3)


ax.plot_surface(x, y, z)

ax.set_xlim3d(0, 2)
ax.set_ylim3d(1, 3)
ax.set_zlim3d(16, 18)
ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z')
ax.view_init(elev = -68, azim=-88)
plt.show()



