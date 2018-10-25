#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:19:00 2018

@author: anshul
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as scisqrt

# initialization of mean and covariance matrix
mu1= np.matrix([[1], [2]])
mu2 = np.matrix([[-1], [1]])
mu3= np.matrix([[2], [-2]])

covar1= np.matrix([[1, 0], [0, 2]])
covar2 = np.matrix([[2, -1.8], [-1.8, 2]])
covar3 = np.matrix([[3, 1], [1, 2]])

N=100

# Creating gaussian distribution
covar1_sqrt = scisqrt.sqrtm(covar1)
covar1_sqrt = np.asmatrix(covar1_sqrt)
covar2_sqrt = scisqrt.sqrtm(covar2)
covar2_sqrt= np.asmatrix(covar2_sqrt)
covar3_sqrt = scisqrt.sqrtm(covar3)
covar3_sqrt= np.asmatrix(covar3_sqrt)

Gauss1= (covar1_sqrt) * np.random.randn(2, N) + mu1
Gauss2= (covar2_sqrt) * np.random.randn(2, N) + mu2
Gauss3= (covar3_sqrt) * np.random.randn(2, N) + mu3
print()



sample_mean1=(1/N) * np.sum(Gauss1, axis=1) # creating mean
xy1= Gauss1[0,:] * np.transpose(Gauss1[1, :]) # creating xy mean
sample_mean_xy1= (1/N)*xy1
G1=Gauss1
xy1_gauss= np.asarray(G1)

sample_mean2=(1/N) * np.sum(Gauss2, axis=1) # creating sample mean
xy2= Gauss2[0,:] * np.transpose(Gauss2[1, :])
sample_mean_xy2= (1/N)*xy2
G2=Gauss2
xy2_gauss= np.asarray(G2)

sample_mean3=(1/N) * np.sum(Gauss3, axis=1)
xy3= Gauss3[0,:] * np.transpose(Gauss3[1, :])
sample_mean_xy3= (1/N)*xy3
G3=Gauss3
xy3_gauss= np.asarray(G3)

mean1= Gauss1 - sample_mean1 # creating xx, yy mean
mean_xy1= (xy1_gauss[0,:]*xy1_gauss[1,:]) - sample_mean_xy1 # creating xy mean
fin_mean_xy1= (1/N)*np.sum(mean_xy1, axis=1)

mean2= Gauss2 - sample_mean2
mean_xy2= (xy2_gauss[0,:]*xy2_gauss[1,:]) - sample_mean_xy2
fin_mean_xy2= (1/N)*np.sum(mean_xy2, axis=1)

mean3= Gauss3 - sample_mean3
mean_xy3= (xy3_gauss[0,:]*xy3_gauss[1,:]) - sample_mean_xy3
fin_mean_xy3= (1/N)*np.sum(mean_xy3, axis=1)

# Creating 2X2 covariance matrix
sample_covar_mat1=(1/(N-1)) * np.sum(np.square(mean1), axis=1)
sample_covar_mat2=(1/(N-1)) * np.sum(np.square(mean2), axis=1)
sample_covar_mat3=(1/(N-1)) * np.sum(np.square(mean3), axis=1)

covariance_mat1=np.matrix([[sample_covar_mat1[0,0], fin_mean_xy1[0,0]],[fin_mean_xy1[0,0],sample_covar_mat1[1,0]]])
w1, v1 = np.linalg.eig(covariance_mat1) # Eigenvalues and eigenvectors
print ("Eigen vectors of Covariance matrix 1 is: ", v1)


covariance_mat2=np.matrix([[sample_covar_mat2[0,0], fin_mean_xy2[0,0]],[fin_mean_xy2[0,0],sample_covar_mat2[1,0]]])
w2, v2 = np.linalg.eig(covariance_mat2)
print ("Eigen vectors of Covariance matrix 2 is: ", v2)

covariance_mat3=np.matrix([[sample_covar_mat3[0,0], fin_mean_xy3[0,0]],[fin_mean_xy3[0,0],sample_covar_mat3[1,0]]])
w3, v3 = np.linalg.eig(covariance_mat3)
print ("Eigen vectors of Covariance matrix 3 is: ", v3)

# Creating initial point as mean
eig_val_sqrt1= np.sqrt(w1)
v1[0, 0] = eig_val_sqrt1[0,]* v1[0, 0]
v1[1, 1] = eig_val_sqrt1[1,] * v1[1, 1]
add_v1= v1 + sample_covar_mat1
new_eig_vect1= np.concatenate((sample_covar_mat1, add_v1), axis=1)

eig_val_sqrt2= np.sqrt(w2)
v2[0, 0] = eig_val_sqrt2[0,]* v2[0, 0]
v2[1, 1] = eig_val_sqrt2[1,] * v2[1, 1]
add_v2= v2 + sample_covar_mat2
new_eig_vect2= np.concatenate((sample_covar_mat2, add_v2), axis=1)

eig_val_sqrt3= np.sqrt(w3)
v3[0, 0] = eig_val_sqrt1[0,]* v3[0, 0]
v3[1, 1] = eig_val_sqrt1[1,] * v3[1, 1]
add_v3= v1 + sample_covar_mat3
new_eig_vect3= np.concatenate((sample_covar_mat3, add_v3), axis=1)

# Ploting the points eigen vectors:


#plt.show()

x1_row1= (1/eig_val_sqrt1[0,]) * (sample_mean1[0, 0]) * mean1[0, 🙂
x1_row2 = (1/eig_val_sqrt1[1,]) * (sample_mean1[1, 0]) * mean1[1, 🙂
new_X_gauss1= np.concatenate((x1_row1, x1_row2), axis=0)


plt.figure()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.scatter([np.transpose(Gauss1[0,:])], [np.transpose(Gauss1[1,:])], marker="^")
plt.plot(np.transpose(new_eig_vect1[0, :2]), np.transpose(new_eig_vect1[1, :2]), color='r', linewidth=2.0)
plt.plot(np.transpose(new_eig_vect1[0, ::2]), np.transpose(new_eig_vect1[1, ::2]), color='b', linewidth=2.0)
plt.scatter([np.transpose(new_X_gauss1[0,:])], [np.transpose(new_X_gauss1[1,:])], marker="o")

x2_row1= (1/eig_val_sqrt2[0,]) * (sample_mean2[0, 0]) * mean2[0, 🙂
x2_row2 = (1/eig_val_sqrt2[1,]) * (sample_mean2[1, 0]) * mean2[1, 🙂
new_X_gauss2= np.concatenate((x2_row1, x2_row2), axis=0)


plt.figure()
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.scatter([np.transpose(Gauss2[0,:])], [np.transpose(Gauss2[1,:])], marker="^")
plt.plot(np.transpose(new_eig_vect2[0, :2]), np.transpose(new_eig_vect2[1, :2]), color='r', linewidth=2.0)
plt.plot(np.transpose(new_eig_vect2[0, ::2]), np.transpose(new_eig_vect2[1, ::2]), color='b', linewidth=2.0)
plt.scatter([np.transpose(new_X_gauss2[0,:])], [np.transpose(new_X_gauss2[1,:])], marker="o")


x3_row1= (1/eig_val_sqrt3[0,]) * (sample_mean3[0, 0]) * mean3[0, 🙂
x3_row2 = (1/eig_val_sqrt3[1,]) * (sample_mean3[1, 0]) * mean3[1, 🙂
new_X_gauss3= np.concatenate((x3_row1, x3_row2), axis=0)


plt.figure()
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.scatter([np.transpose(Gauss3[0,:])], [np.transpose(Gauss3[1,:])], marker="^")
plt.plot(np.transpose(new_eig_vect3[0, :2]), np.transpose(new_eig_vect3[1, :2]), color='r', linewidth=2.0)
plt.plot(np.transpose(new_eig_vect3[0, ::2]), np.transpose(new_eig_vect3[1, ::2]), color='b', linewidth=2.0)
plt.scatter([np.transpose(new_X_gauss3[0,:])], [np.transpose(new_X_gauss3[1,:])], marker="o")
