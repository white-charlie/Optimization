#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:34:06 2024

@author: charliewhite
"""

#Question 1

import numpy as np
import matplotlib.pyplot as plt

#Create an array of the locations of each of the factories
factories = np.array([[1,1],[1,3],[2,5],[3,1]])

#Set weights equal to 1 for each factory
weights = np.ones(len(factories))

#Define the gradient descent function
def grad_descent(factories, weights, x0, tol=1e-5):
    
    x_k = np.array(x0)
    iterations = 0
    trajectory = [x_k.copy()]
    
    while True:
        grad = np.zeros_like(x_k)
        denom = 0
        
        
        for i in range(len(factories)):
            difference = x_k - factories[i]
            norm = np.linalg.norm(difference)
            if norm > 0:
                
                #Calculate gradient and stepsize as seen in lectures
                grad += weights[i] * (difference / norm)
                denom += weights[i] * 1/norm
        
        #Set stopping criteria
        if np.linalg.norm(grad) < tol:
            break
        
        #Set up the iterative process
        t_k = 1/denom    
        x_k = x_k - t_k * grad
        trajectory.append(x_k.copy())
        iterations += 1
   
    return x_k, iterations, np.array(trajectory)

# Initial guess
x0 = np.array([0.0,0.0])

new_x, iterations, trajectory = grad_descent(factories, weights, x0)


#Plot the convergence
plt.figure(figsize=(12,10))
plt.plot(trajectory[:,0], trajectory[:,1], 'o-',color='r', label='Convergence Path')
plt.scatter(factories[:,0], factories[:,1], color='b', label='Factories', s=80)
labels = ['A', 'B', 'C', 'D']
for i, txt in enumerate(labels):
    plt.text(factories[i,0]+0.01,factories[i,1]+0.01, txt, fontsize=25)
plt.scatter(trajectory[-1,0], trajectory[-1,1], color='g', label='Optimal location',zorder=2)
plt.title('Convergence of Gradient Descent for the Fermat-Weber problem')
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
plt.legend()
plt.grid(True)
plt.plot()

print("Location:", new_x)
print("Number of iterations:", iterations)

#%%

#vi)
centroid = np.mean(factories, axis=0)
print(centroid)
new_x, iterations, trajectory = grad_descent(factories, weights,centroid)
print("Location:", new_x)
print("Number of iterations:", iterations)

#%%

#b)

weights_new = np.array([1.5,1,1,1])
new_x, iterations, trajectory = grad_descent(factories, weights_new, x0)
print("Location:", new_x)
print("Number of iterations:", iterations)

#Plot the convergence
plt.figure(figsize=(12,10))
plt.plot(trajectory[:,0], trajectory[:,1], 'o-',color='r', label='Convergence Path')
plt.scatter(factories[:,0], factories[:,1], color='b', label='Factories', s=80)
for i, txt in enumerate(labels):
    plt.text(factories[i,0]+0.01,factories[i,1]+0.01, txt, fontsize=25)
plt.scatter(trajectory[-1,0], trajectory[-1,1], color='g', label='Optimal location',zorder=2)
plt.title('Convergence of Gradient Descent for the Fermat-Weber problem')
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
plt.legend()
plt.grid(True)
plt.plot()

#%% Question 2


def Newtons_method(x_0, tol):
    
    #Initialise the array x_k to be our initial guess x_0
    x_k = np.array(x_0)
    iterations = 0
    
    while True:
        
        #Directly compute first order derivatives
        dx = -200*(x_k[3]**2 - x_k[0]) + 20.2*(x_k[0]-1) + 19.8*(x_k[2]-1)
        dy = 2*(x_k[1]-1) +360*x_k[1]*(x_k[1]**2 - x_k[2])
        dz = -180*(x_k[1]**2 - x_k[2]) + 20.2*(x_k[2]-1) + 19.8*(x_k[0]-1)
        dw = 400*x_k[3]*(x_k[3]**2-x_k[0]) + 2*(x_k[3]-1)
        
        #Define the gradient vector
        grad = np.array([[dx],[dy],[dz],[dw]])
        
        #Directly define second order derivatives
        dxx = 220.2
        dyy = 2 + 1080*(x_k[1]**2) - 360*x_k[2]
        dzz = 200.2
        dww = 1200*(x_k[3]**2) - 400*x_k[0]+2
        dxy = dyx = dyw = dzw = dwy = dwz = 0
        dxz = dzx = 19.8
        dxw = dwx = -400*x_k[3]
        dyz = dzy = -360*x_k[1]
        
        #Define the Hessian
        hessian = np.array([[dxx,dxy,dxz,dxw],
                        [dyx,dyy,dyz,dyw],
                        [dzx,dzy,dzz,dzw],
                        [dwx,dwy,dwz,dww]])
        
        #Compute d_k by solving the linear system
        try:
            d_k = np.linalg.solve(hessian, -grad)
        except np.linalg.LinAlgError:
            print("The Hessian is ill-conditioned")
            break
        
        #Set up the iterative process
        x_k = x_k + d_k.flatten()
        iterations += 1
    
        if np.linalg.norm(grad) <= tol:  
            break
        
    return x_k, iterations

x_0 = np.array([0,1,2,3])

new_x, iterations = Newtons_method(x_0, 1e-5)
print("Solution:", new_x)
print("Number of iterations:", iterations)


