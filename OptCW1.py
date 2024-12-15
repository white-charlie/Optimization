#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:43:47 2024

@author: charliewhite
"""

#%% Question 1
#a)

import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = np.loadtxt('/Users/charliewhite/Downloads/t_data_x_noisy.csv', delimiter =',', skiprows =1)

# Split the columns into time and noisy data
t_data = data [:, 0]
x_noisy = data [:, 1]

from scipy.interpolate import UnivariateSpline

# Fit the noisy data using a spline to approximate x'(t)
spline_fit = UnivariateSpline (t_data , x_noisy , s=0.5) # Smoothing spline
spline_derivative = spline_fit . derivative () # Derivative of the spline

# Get the derivative values from the spline approximation
x_prime_approx = spline_derivative (t_data) # This is the approximation of x'(t)

#Create a function to calculate the matrix D
def compute_D(n):
    D = np.zeros((n+1, n+1))
    
    for i in range(2, n+1):
        D[i, i] = i * (i - 1)
    return D

def regularised_LS(lambda_k,n):
    
    # Create a Vandermonde matrix (A) with 1's in the left hand column
    A = np.vander(t_data, n+1, True)

    # Set b as the already approximated values of x'(t) using x_prime_approx
    b = x_prime_approx
  
    #Call the function compute_D
    D = compute_D(n)
            
    # Use the regularised least squares result from lectures to compute c 
    c = np.linalg.inv(A.T @ A + lambda_k* D.T @ D) @ A.T @ b
    
    return c

def f(lambda_k, n, x):    

#Define f(x) by reversing the coefficients in c and evaluating them at x
    return np.polyval(regularised_LS(lambda_k,n)[::-1], x)

print(regularised_LS(0,5))

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(10, 8))

ax1.plot(t_data, x_noisy, label='Noisy data',color='g')
ax1.scatter(t_data, spline_fit(t_data), label='Spline approximation of x(t)', color='r')
ax1.set_title('Noisy Data and Spline fit of x(t)')
ax1.set_xlabel('t_data')
ax1.set_ylabel("x(t)")
ax1.legend()

ax2.scatter(t_data, x_prime_approx, label="Spline approximation of x'(t)", color='orange')
ax2.set_title("Spline Approximation of x'(t)")
ax2.set_xlabel("t_data")
ax2.set_ylabel("x'(t)")
ax2.legend()

ax3.scatter(t_data, x_prime_approx, label="Spline approximation of x'(t)", color='orange')
ax3.scatter(t_data, f(0.1,5,t_data),s=6, label="Polynomial approximation of x'(t)", color='b')
ax3.set_title("Polynomial Approximation of tx'(t)")
ax3.set_xlabel("t_data")
ax3.set_ylabel("x'(t)")
ax3.legend()

plt.show()

#%%
#b)

#Generate 100 evenly spaced numbers over [0,0.25] to 
t_real=np.linspace(0,2.5,100)

# Define the true derivative function x'(t)
def x_prime_real(t_real):
    return -np.exp(-t_real)

plt.figure(figsize=(10,8))
plt.plot(t_real,x_prime_real(t_real),label="True derivative", color='g', linewidth=3)
plt.scatter(t_data, f(0,5,t_data),s=3,label="Lambda = 0", color='darkorange')
plt.scatter(t_data, f(1,5,t_data),s=3,label="Lambda = 1", color='blueviolet')
plt.scatter(t_data, f(100,5,t_data),s=3,label="Lambda = 100", color='fuchsia')
plt.scatter(t_data, f(10000,5,t_data),s=3,label="Lambda = 10000", color='crimson')
plt.title("Polynomial approximations of x'(t)")
plt.xlabel('t data')
plt.ylabel("x'(t)")
plt.legend()
plt.show()

def condition_number(lambda_k,n):
    
    A = np.vander(t_data, n+1, True)
    D = np.zeros((n+1, n+1))
    
    for i in range(2, n+1):
        D[i, i] = i * (i - 1)
        
    #Calculate the design matrix named normal_matrix
    normal_matrix = A.T @ A + lambda_k* (D.T @ D)

    return np.linalg.cond(normal_matrix)

lambda_values = [0,1,100,10000]

for l in lambda_values:
    cond = condition_number(l,5)
    print(f'For lambda = {l}, condition number = {cond}\n')


#%% c)

''' 
'''

#%% Question 2

#a)

#Create an array of t-values as in the given dataset
t_i = np.array([1,2,4,8,16,24,32,48,64,96,128,192,256,384])
Y_i = ([77.5,136.2, 240.4, 356.0, 505.6, 643.2, 700.8, 720.0, 710.4, 748.8, 666.0, 614.4, 588.8, 153.6])

#Create vectors x_1,x_2,x_3
x_1 = np.array([200,0.3,0.5])
x_2 = np.array([120,0.25,0.5])
x_3 = np.array([180,0.2,0.7])

#Define the function f to return Y_i
def f(t,x):   
    return x[0]*t*np.exp(-x[1]*t**x[2])


#Plot a scatterplot of the data
plt.figure(figsize=(10,8))

plt.scatter(t_i,f(t_i,x_1),label="x_1 = [200,0.3,0.5]", color='darkorange' )
plt.plot(t_i,f(t_i,x_1), color='darkorange' )

plt.scatter(t_i, f(t_i,x_2),label="x_2 = [120,0.25,0.5]", color='blueviolet')
plt.plot(t_i,f(t_i,x_2), color='blueviolet' )

plt.scatter(t_i, f(t_i,x_3),label="x_3 = [180,0.2,0.7]", color='crimson')
plt.plot(t_i,f(t_i,x_3), color='crimson' )

plt.scatter(t_i, Y_i,s=45, label='Observed Data',color ='g')

plt.grid(True)
plt.xlabel('Number of first generation weevil couples (t_i)')
plt.ylabel('Average number of adult weevils (Y_i)')
plt.title('Changes in Population Density of Weevils')
plt.legend()
plt.show()

#%%
#b)

#Define a function to construct the Jacobian matrix
def Jacobian(t,x):
    
    J_xk = []
    
    for t_i in t:
        
        #Explicitly calculate the partial derivatives of f with respect to x1,x2,x3 respectively
        pderiv_x1 = t_i*np.exp(-x[1]*t_i**x[2])
        pderiv_x2 = -x[0]*(t_i**(1+x[2]))*np.exp(-x[1]*t_i**x[2])
        pderiv_x3 = -x[0]*x[1]*t_i**(1+x[2])*np.log(t_i)*np.exp(-x[1]*t_i**x[2])
        
        J_xk.append([pderiv_x1, pderiv_x2, pderiv_x3])
    
    return np.array(J_xk)
        
x_vectors = [x_1,x_2,x_3]
for x_i in x_vectors:
    J_xk = Jacobian(t_i,x_i)    
    print(f"Jacobian with x vector {x_i}:\n {J_xk}")
 
#%% 
#c)

#Define F(x) using f(t_i,x) values and observed Y_i values
def F(t,x,Y): 
    return f(t,x)-Y      

#Define the gradient function of g using the Jacobian and F(X)
def grad_g(t,x,Y):
    
    J = Jacobian(t,x)
    F_x = F(t,x,Y)
        
    return 2*J.T@F_x

#Create the Gauss-Newton function
def Gauss_Newton(t,x,Y,tol):
    
    #Set iterations to 0
    iterations = 0

    #Set the stopping criteria 
    while np.linalg.norm(grad_g(t,x,Y)) > tol:
        
        J = Jacobian(t,x)        
        JTJg = np.linalg.inv(J.T @ J) @ grad_g(t,x,Y)
        
        #Iterate using the Gauss-Newton result from lectures
        x = x - 0.5*JTJg  
        
        #Increase iterations by 1
        iterations+=1
        
    return x, iterations

x_vectors = [x_1,x_2,x_3]
for x_i in x_vectors:
    x_GN, iterations = Gauss_Newton(t_i,x_i,Y_i,1e-6)    
    print(f"Gauss-Newton with initial vector {x_i}:\n Solution = {x_GN}, \n Iterations ={iterations}")
        

#%% 
#d)

def damped_Gauss_Newton(t, step,x,Y,tol):
    
    iterations_damped = 0
    
    while np.linalg.norm(grad_g(t,x,Y)) > tol:
        
        #Call the Jacobian and F(x) functions
        J = Jacobian(t,x)
        F_x = F(t,x,Y)
        
        #Define d_k as in the lecture notes
        d_k = -np.linalg.inv(J.T @ J) @(J.T @ F_x) 
        
        #Iterate using a set step_size (step) and d_k as defined above
        x = x + step*d_k
        
        #Increase iterations by 1
        iterations_damped += 1
        
    return x, iterations_damped
        
step_values = [0.8,0.6,0.4]
for s in step_values:
    x_damped, iterations_damped = damped_Gauss_Newton(t_i, s, x_3,Y_i,1e-6)
    print(f"Damped Gauss-Newton with step-size = {s}:\n Solution = {x_damped}, \n Iterations ={iterations_damped}")  
    
        




