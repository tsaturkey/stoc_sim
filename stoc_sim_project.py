# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:58:47 2023

@author: Lara Turgut
"""

import time
import random
import winsound
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


"""
 Define the collision kernels 
"""

def K1(x, y, _):
    return np.where((x * y > 0), 1, 0)

def K2(x, y, _):
    return x*y 

def K3(x, y, _):
    return np.where((x * y > 0), (x + y) , 0)

def K4(x, y, _):
    return np.where((x * y > 0), 0.25 * np.power((np.power(x, 1/3) + np.power(y, 1/3)), 3), 0)
    #return 0.25 * np.power((np.power(x, 1/3) + np.power(y, 1/3)), 3) if (x * y > 0) else 0

# For the task 3
def K_a(x, y, a):
    return x**a * y**a


"""

  Calculate the *total* collision rate for a given state x and a given kernel
  This total collision rate (i.e. sum of the kernels K(x_i,x_j) for i!=j) is nessesary for the 
  implementation of the index distribution (sampling) and for the simulation (time evolution).
  
      ************************************
      
   My old code with nested for-loops (I switsched to meshgrid later on for performace):
       
       def total_collision_rate(x, kernel, a):
           
           N = len(x)  # number of particles
           total_rate = 0

           for i in range(N):
               for j in range(N):
                   if i != j:
                       total_rate += kernel(x[i], x[j], a)

           return total_rate

      ***************************************
"""


def total_collision_rate(x, kernel, a):
    
    mesh_x1, mesh_x2 = np.meshgrid(x, x)
    pairs = np.stack([mesh_x1, mesh_x2], axis = -1)
    collision_rates = kernel(pairs[:, :, 0], pairs[:, :, 1], a)
    
    return np.sum(collision_rates) - np.sum(kernel(np.diag(x), np.diag(x), a)), collision_rates




"""
  Index distribution sampled using the discrete inverse-transform method
  Input: Kernel values K(x_i, x_j) for i!=j and cumulative probabilities
  
  *****************************************
    My old code with for-loops: 
        
    def index_distributionn(kernel, x, N, total_rate, a):
            
        U = random.uniform(0, 1)
        
        cumulative_probability = 0
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    probability = kernel(x[i], x[j], a) / total_rate
                    cumulative_probability += probability
                    
                    if (U < cumulative_probability):
                       return (i, j)
"""

def index_distribution(kernel, x, N, total_rate, kernel_values, a):
    U = np.random.uniform(0, 1)
    
    np.fill_diagonal(kernel_values, 0)    
    
    # Compute cumulative probabilities
    cumulative_probability = np.cumsum(kernel_values.flatten() / total_rate)

    # Find the first index where U < cumulative probability
    index = np.argwhere(U < cumulative_probability)[0][0]

    # Convert the flattened index back to i and j
    i_index = index // N
    j_index = index % N

    return i_index, j_index


              
                    
"""
    This function simulates the coagulation process in the time interval [0,T].
    
    We start with the initial size state x_0. Our time steps (holding times) are sampled 
    from an exponential distribution for independence (due to the homogeneity of the Markov chain).
    
    The particles to be coagulated are sampled from the index_distribution function.
    
"""            


def simulate_coagulation(kernel, x_0, T, a):
    
    # x is the current particle-size state
    # set the initial sizes
    x = list(x_0)
    # initial time, zero point
    current_time = 0
    
    # Generate lists for recording the simulation steps
    # I.e., these lists will record the time evolution of the particle sizes x 
    # as a function of t.
    t_simulation = [] 
    t_simulation.append(0) # the starting time is 0
    x_simulation = []
    x_simulation.append(list(x_0)) # the initial state is x_0

    # start the time evolution/simulation
    while current_time < T: 

        total_rate, kernel_values = total_collision_rate(x, kernel, a)
        
        # if the sum of the transitions probabilities is zero, we won't be getting coagulation anymore.
        # the total rate being zero means that all particles coagulated together.
        if total_rate == 0:
            break
        
    
        # Sample the waiting time until the next collision event
        # Draw samples from an exponential distribution (see cont. time/discrete state Markov chain in Ch. 4.7)
        # The holding time depends only on the size of the particles, i.e. the process is homogeneous.
        holding_time = random.expovariate(total_rate / (2*N))
        current_time += holding_time
        
        if current_time > T:
            break
        
        # record the current time
        t_simulation.append(current_time)
        
        # sample from the index distribution, i.e. pick i and j to coagulate.
        i, j = index_distribution(kernel, x, N, total_rate, kernel_values, a)
        
        
        # coagulate i and j
        # to coagulate i and j means: x_i -> x_i + x_j and x_j -> 0
        x[i] = x[i] + x[j]
        x[j] = 0
        
        # record the current state
        x_simulation.append(list(x))
        

    return x_simulation, t_simulation


"""
    This function approximates the average particle concentration for a simulation.
    
    x_simulation includes the simulation steps of the size state x. 
    
    ************************************
    
    My old code with nested for-loops :
        
    def compute_avg_concentration(N, x_simulation, k):
        
        num_step = len(x_simulation) # this gives the number of steps in the simulation x_simulation.
        avg_c = []
        
        
        for t in range(num_step):
            sum_size_k = 0
            for x_i in x_simulation[t]:
                if x_i == k:
                    sum_size_k += 1
            avg_c.append(sum_size_k / N)
                
        return avg_c
    
"""
     

def compute_avg_concentration(N, x_simulation, k):
    x_simulation = np.array(x_simulation)
    
    mask = (x_simulation == k)[:,:, np.newaxis]
    
    avg_c = np.sum(mask, axis=1) / N
    return avg_c



def compute_moments(N, x_simulation, p):
    
    t_points = len(x_simulation)
    moments = []
    
    for t in range(t_points):
        moments.append(np.mean(np.power(x_simulation[t], p)))
    return moments



def compute_confidence_intervals(R, results, T):
    
    # Calculate the mean concentration from R repetitions in the end of the time interval [0, T]
    mean = np.mean(results)
    
    # Calculate the standard error for the mean concentration
    std_err = np.std(results)
    
    # Set a significance level (e.g., 0.05) and calculate the critical value
    alpha = 0.05
    cval = st.norm.ppf(1-alpha/2)
    
    # Calculate the confidence interval for the mean concentration
    lower_bound = mean - cval * std_err / np.sqrt(R)
    upper_bound = mean + cval * std_err / np.sqrt(R)
    
    return lower_bound, upper_bound   
 
   


if __name__=='__main__':
    
    N = 10000 # Number of particles
    T = 10  # End time of simulation
    R = 10  # Number of simulations
    p = 1.5
    k_values = [1, 5, 15, 50]
    a_values = [0.7, 0.8, 0.9, 1]
    
    # Mono-dispersal initial condition
    # x_0 is the state of the initial system consisting of N particles.
    x_0 = [1] * N


    # repeat the simulation R times to compute condidence intervals for the average concentration and moment
    
    """
    avg_concentrations_r = [] #average concentrations at time T for R different simulations
    moments_r = []            #moments at time T for R different simulations
    
    
    for r in range(R):
        x_simulation, _ = simulate_coagulation(K4, x_0, T, a_values[1])
        avg_concentrations_r.append(compute_avg_concentration(N, x_simulation, k_values[0])[-1])
        moments_r.append(compute_moments(N, x_simulation, p)[-1])
        
    confidence_interval_avg_c = compute_confidence_intervals(R, avg_concentrations_r, T)
    confidence_interval_moments = compute_confidence_intervals(R, moments_r, T)
    
    """ 
    
    # Simulate
    tic = time.perf_counter()
    
    x_simulation, t_simulation = simulate_coagulation(K3, x_0, T, a_values[0])
    
    toc = time.perf_counter()
    
    print("Simulation time:" )
    print(toc-tic)
    
    # Plot average concentration for all k values in the list k_values
    
    for i, k in enumerate(k_values):
        plt.figure(figsize=(12, 6))
        plt.plot(t_simulation, compute_avg_concentration(N, x_simulation, k))
        plt.xlabel('Time')
        plt.ylabel(f'C(t, {k})')
        plt.title(f'Average concentration for k = {k}')
        

    # Plot the moment for p =1.5
    
    plt.figure(figsize=(12, 6))
    plt.plot(t_simulation, compute_moments(N, x_simulation, p))
    plt.xlabel('Time')
    plt.ylabel(f'm(t)')
    plt.title(f'Moment for p = 1.5')
    
    frequency = 2500  # Set frequency (2500 Hz)
    duration = 1000  # Set duration (1000 ms = 1 second)
    winsound.Beep(frequency, duration)
    
    
   