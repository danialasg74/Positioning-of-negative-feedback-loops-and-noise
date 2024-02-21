import random
import numpy as np
import numba as nb
from numba import jit
import json
import numpy as np
import math
import random
from numba import jit
import numba as nb
from itertools import product
import os
import json



#Noise = sd/mean    
def calculate_noise(std_vector, aver_traj_vector):
    # Create an array to store the noise values
    noise_vector = np.zeros_like(std_vector)
    
    # Calculate the noise for each index
    for i in range(len(std_vector)):
        if aver_traj_vector[i] != 0:
            noise_vector[i] = std_vector[i] / aver_traj_vector[i]
        else:
            noise_vector[i] = 0
    
    return noise_vector

#the length of each side of the lattice
side = 100


@jit(nopython=True)
def Bacterial_sim(i0, j0, lattice,p , nbacteria):
    # Coordinate of the starting point
    i = i0
    j = j0
    
    # Counter to keep track of the number of bacteria
    num_bacteria = 0
    
    # Repeat the loop until nbacteria is added to the lattice
    while num_bacteria < nbacteria:
        # Choose a direction randomly (up, down, right, or left)
        val = random.randint(1, 4)
        
        # For right:
        if val == 1 and i < side - p:
            i = i + p
            j = j
        # Boundary condition for right side of the lattice:
        if val == 1 and i == side - p:
            i = 0
            j = j
        # For left:
        if val == 2 and i >= p:
            i = i - p
            j = j
        # Boundary condition for the left side of the lattice:
        if val == 2 and i < p:
            i = side - 1
            j = j
        # For up:
        if val == 3 and j < side - p:
            i = i
            j = j + p
        # Boundary condition for the top of the lattice:
        if val == 3 and j == side - p:
            i = i
            j = 0
        # For down:
        if val == 4 and j >= p:
            i = i
            j = j - p
        # Boundary condition for the bottom of the lattice:
        if val == 4 and j < p:
            i = i
            j = side - 1
        if lattice[i, j] !=1:
            num_bacteria += 1
            
        # Place a bacteria at (i, j) coordinate
        lattice[i, j] = 1
        
        # Increment the bacteria counter
        
        




#Events in the sim
from numba import jit
events0=[(0,0,0,0,0,0,0)]

events=[
 #B#R#C#N#P#S#A   
(+1,0,0,0,0,0,0),
(-1,0,0,0,0,0,0),
    
(0,+1,0,0,0,0,0),
(0,-1,1,0,0,0,0),
(0,-1,0,0,0,0,0),

(0,0,-1,0,0,0,0),
    
(0,0,0,+1,0,0,0),
(0,0,0,-1,0,0,0),
    
(0,0,0,0,+1,0,0),  
(0,0,0,0,-1,0,0),
    
(0,0,0,0,0,+1,0), 
(0,0,0,0,0,-1,0),
    
(0,0,0,0,0,0,+1),
(0,0,0,0,0,0,-1)
]

events = np.asarray(events)
events0 = np.asarray(events0)

#Gillespie algorithm
@jit(nopython=True)
def Gillespie(size, Input , k0 , b1 , b2, b3, b4, b5, q1, Zn, Zs, 
B1,
R1,
C1,
N1,
P1,
S1,
A1):
    
    #Intializing time steps
    ns   = 0
    time = 0
    
    #Number of events
    no_events = np.arange(0,14,1)
    
    #Vector of t and proteins
    t = np.zeros(size)
    v = np.zeros((size, 7))
    
    #Initial conditions
    v[(0, 0)] = B1
    v[(0, 1)] = R1
    v[(0, 2)] = C1
    v[(0, 3)] = N1
    v[(0, 4)] = P1
    v[(0, 5)] = S1
    v[(0, 6)] = A1
    
    #Run the sim until maxtime
    for ns in range(size-1):
        
        if Input[ns] ==1:
            v[(ns, 0)] = 1
        #Rates of reactions
        rates = np.array([
                          k0*v[(ns, 0)], #k0*B
                          v[(ns, 6)]*v[(ns, 0)], #A*B
                          
                          
                          b1* ( v[(ns, 3)] / ( v[(ns, 3)] + Zn) ), #b1 (Nt/ (Nt + Zn))
                          v[(ns, 0)]*v[(ns, 1)], #B*R
                          v[(ns, 4)]*v[(ns, 1)], #P*R
                          
                          v[(ns, 4)]*v[(ns, 2)],#P*C
                          
                          b2*v[(ns, 2)],#b2*C
                          q1*v[(ns, 3)],#q1*N
                          
                          b3 * ( v[(ns, 3)] / ( v[(ns, 3)] + Zn) ),#b3 (Nt/ (Nt + Zn))
                          q1*v[(ns, 4)],#q1*P
                          
                          b4 * ( v[(ns, 3)] / ( v[(ns, 3)] + Zn) ),#b4 (Nt/ (Nt + Zn))
                          q1*v[(ns, 5)],#q1*S
                          
                          b5 * ( v[(ns, 3)] / ( v[(ns, 3)] + Zn + (v[(ns, 5)] * Zn / Zs)  ) ),#b5 (Nt/ (Nt + Zn + (S*Zn /Zs) ))
                          q1*v[(ns, 6)] #q1*A
                          
                         ])
        
        
        
        #Total rate
        total_rate  =  np.sum(rates)
        
        if total_rate == 0:
            v[ns+1] = v[ns]+events0[0]
            ns = ns + 1
        else:

            choose = np.searchsorted(np.cumsum(rates/total_rate), np.random.rand(1))[0]
            
            v[ns+1] = v[ns]+events[choose]
            ns = ns + 1
         
    return v[0:size]



#use numba to speed up simulations
@jit(nopython=True)
def walk(size):
        #Go for a random walk (in the same environment)
        i = round(random.uniform(0, m-1))
        j = round(random.uniform(0, m-1))
        Input = np.zeros(size)
    
        for z in range(size):
            val = random.randint(1, 4)
            if val == 1 and i <m-1:
                i = i + 1
                j = j
            if val == 1 and i ==m-1:
                i = 0
                j = j
            if val == 2 and i >0:
                i = i - 1
                j = j
            if val == 2 and i ==0:
                i = m-1
                j = j
            if val == 3 and j <m-1:
                i = i
                j = j + 1
            if val == 3 and j ==m-1:
                i = i
                j = 0
            if val == 4 and j>0:
                i = i
                j = j - 1
            if val == 4 and j ==0:
                i = i
                j = m-1
            Input[z] = lattice[i,j]
        return Input
    
    
    

from itertools import product
b1_vec = np.arange(1, 11, 1)
b2_vec = np.arange(1, 11, 1)
b5_vec = np.arange(1, 11, 1)
q_vec  = np.arange(0.1, 1.1, 0.1)
combos_background = np.round(list(product(b1_vec, b2_vec, b5_vec, q_vec)),1)



size = 1000
m = 100
iterations = 10000
k0 = 0.1
b3 = 1
b4 = 1
Zn = 1
Zs = 1
B1 = 0
R1 = 10
C1 = 0
N1 = 0
P1 = 0
S1 = 0
A1 = 0



for jj in range(len(combos_background)):
    
    b1 = combos_background[jj][0]
    b2 = combos_background[jj][1]
    b5 = combos_background[jj][2]
    q1 = combos_background[jj][3]
    
    
    #Make a lattice of size 10,000
    lattice = np.zeros((side,side))
    
    #Find a random starting point for simulation of the bacterial population (i0,j0)
    i0=round(random.uniform(0, side-1))
    j0=round(random.uniform(0, side-1))
    
    #Simulate the bacterial population
    nbacteria = 200
    Bacterial_sim(i0,j0,lattice, 5, nbacteria)
    v_vector = []
    for _ in range(iterations):
        
        #individuals have their own trajectory walks
        Input = walk(size)

        v =  Gillespie(
                  size = size,  
                  Input=Input, 
                  k0 = k0, 
                  b1 = b1, 
                  b2 = b2, 
                  b3 = b3, 
                  b4 = b4, 
                  b5 = b5, 
                  q1 = q1, 
                  Zn = Zn, 
                  Zs = Zs,
                  B1 = B1,
                  R1 = R1,
                  C1 = C1,
                  N1 = N1,
                  P1 = P1,
                  S1 = S1,
                  A1 = A1)
        v_vector.append(v)
        
    #Initialize vectors for average values
    vector_size = iterations
    aver_traj_B = 0;aver_traj_R = 0;aver_traj_C = 0;aver_traj_N = 0;aver_traj_P = 0;aver_traj_S = 0;aver_traj_A = 0
    
    #Calculate the average 
    for i in range(iterations):
        
        aver_traj_B += v_vector[i][:,0] / vector_size
        aver_traj_R += v_vector[i][:,1] / vector_size
        aver_traj_C += v_vector[i][:,2] / vector_size
        aver_traj_N += v_vector[i][:,3] / vector_size
        aver_traj_P += v_vector[i][:,4] / vector_size
        aver_traj_S += v_vector[i][:,5] / vector_size
        aver_traj_A += v_vector[i][:,6] / vector_size
    
    dev_B = 0;dev_R = 0;dev_C = 0;dev_N = 0;dev_P = 0;dev_S = 0;dev_A = 0
    
    #Calculate sum of deviations from mean
    for i in range(iterations):
        dev_B += (v_vector[i][:,0] - aver_traj_B)
        dev_R += (v_vector[i][:,1] - aver_traj_R)
        dev_C += (v_vector[i][:,2] - aver_traj_C)
        dev_N += (v_vector[i][:,3] - aver_traj_N)
        dev_P += (v_vector[i][:,4] - aver_traj_P)
        dev_S += (v_vector[i][:,5] - aver_traj_S)
        dev_A += (v_vector[i][:,6] - aver_traj_A)
    
    #Calculate std
    std_B = np.sqrt(   ( (dev_B**2) / vector_size)  )
    std_R = np.sqrt(   ( (dev_R**2) / vector_size)  )
    std_C = np.sqrt(   ( (dev_C**2) / vector_size)  )
    std_N = np.sqrt(   ( (dev_N**2) / vector_size)  )
    std_P = np.sqrt(   ( (dev_P**2) / vector_size)  )
    std_S = np.sqrt(   ( (dev_S**2) / vector_size)  )
    std_A = np.sqrt(   ( (dev_A**2) / vector_size)  )
        
        
    # Calculate noise for each protein
    Noise_B = calculate_noise(std_B, aver_traj_B)
    Noise_R = calculate_noise(std_R, aver_traj_R)
    Noise_C = calculate_noise(std_C, aver_traj_C)
    Noise_N = calculate_noise(std_N, aver_traj_N)
    Noise_P = calculate_noise(std_P, aver_traj_P)
    Noise_S = calculate_noise(std_S, aver_traj_S)
    Noise_A = calculate_noise(std_A, aver_traj_A)
    
    #Calculate fitness for 10,000 individuals
    fitness_vec1 = []
    fitness_vec2 = []
    fitness_vec3 = []
    fitness_vec4 = []
    for i in range(iterations):
        fitness_vec1.append( np.exp(-(np.mean(v_vector[i][:,0]) +np.mean(v_vector[i][:,1]) +np.mean(v_vector[i][:,3]) +np.mean(v_vector[i][:,4]) +np.mean(v_vector[i][:,5]) +np.mean(v_vector[i][:,6]) )) )
        fitness_vec2.append( np.exp(-(np.mean(v_vector[i][:,0])  +np.mean(v_vector[i][:,6]) )) )
        fitness_vec3.append( np.exp(-(np.mean(v_vector[i][:,0]) )) )
        fitness_vec4.append( np.exp(-(np.mean(v_vector[i][:,6]) )) )
        
        
    #Save the results 
    data = {
    
     'time': size,
     'k0': k0,
     'b1': b1,
     'b2': b2,
     'b3': b3,
     'b4': b4,
     'b5': b5,
     'q1': q1,
     'Zn': Zn,
     'Zs': Zs,
     'B1': B1,
     'R1': R1,
     'C1': C1,
     'N1': N1,
     'P1': P1,
     'S1': S1,
     'A1': A1,
        
    'mean_B': aver_traj_B.tolist(),
    'mean_R': aver_traj_R.tolist(),
    'mean_C': aver_traj_C.tolist(),
    'mean_N': aver_traj_N.tolist(),
    'mean_P': aver_traj_P.tolist(),
    'mean_S': aver_traj_S.tolist(),
    'mean_A': aver_traj_A.tolist(),
        
            'sd_B': std_B.tolist(),
            'sd_R': std_R.tolist(),
            'sd_C': std_C.tolist(),
            'sd_N': std_N.tolist(),
            'sd_P': std_P.tolist(),
            'sd_S': std_S.tolist(),
            'sd_A': std_A.tolist(),
        
       'noise_B': Noise_B.tolist(),
       'noise_R': Noise_R.tolist(),
       'noise_C': Noise_C.tolist(),
       'noise_N': Noise_N.tolist(),
       'noise_P': Noise_P.tolist(),
       'noise_S': Noise_S.tolist(),
       'noise_A': Noise_A.tolist(),
        
       'Fitness1': fitness_vec1,
       'Fitness2': fitness_vec2,
       'Fitness3': fitness_vec3,
       'Fitness4': fitness_vec4
    }
    
    #the file name
    file_name = f"data_{jj}.json"
    
    #Save data as JSON
    with open(file_name, 'w') as json_file:
        json.dump(data, json_file)


    
    