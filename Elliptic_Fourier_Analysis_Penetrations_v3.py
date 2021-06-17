# -*- coding: utf-8 -*-
"""
Created on Mon May  6 08:43:28 2019

@author: Tim Gorman
This program is for calculating the elliptic coefficients of an Elliptic Fourier Series based on a set of (x,y) data.
The key aspect of the algorithm is that it limits the highest frequency based on the largest gap between data points.

Updates:
    06/10/2019 - This version can import X,Y data produced by Alex Portolese.  Then that data can be analyzed with
    a variable number of data points.  The goal of this new code is to be able to determine the minimum number of points
    needed to model the WEC penetrations accurately.
"""
#%%
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sympy as sp
import os

#Manually set path to where files are located
os.chdir("//lspt.local/redirected/redir_data/tgorman/Documents/N-080 Westinghouse/Penetration Modeling")

#intialize x, y, and radius data arrays
xa = np.empty([903,12], dtype = float) # x data
ya = np.empty([903,12], dtype = float) # y data
ra = np.empty([903,12], dtype = float) # r data

#load text files
for i in np.arange(0,12,1):
    filename = "Slice" + str(i+1) + ".csv"
    xa[:,i], ya[:,i]=np.loadtxt(fname = filename,skiprows=1, delimiter = ',', unpack = True)
    
# calculating theta angles from (x,y) data
thetas=np.arctan2(ya,xa)
thetas[thetas<0]+=2*np.pi

#  Plot Raw Data
fig, ax = plt.subplots()

for i in np.arange(0,12,1):
    
    #ax.plot(x_model,y_model,linestyle='none',marker='o',label='Model')
    ax.plot(xa[:,i],ya[:,i],linestyle='none',marker='x',markersize='10',label='Test Data')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
ax.legend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    
#%% Calculation of points and theta matrix.  "Theta matrix" is a short hand hame for the final the matrix in the Elliptic Fourier
#analysis that contains all of the cos(theta) and sin(theta) terms.  See my PowerPoint on EFA.

# clearing plots
plt.close("all")
# total points in experimental data
total_points = len(xa)
# setting up angular sampling rates
sampling_ranges=np.arange(1,12,1);

#initializing error arrays
x_error = np.zeros(shape=(xa.shape[1],len(sampling_ranges)))
y_error = np.zeros(shape=(xa.shape[1],len(sampling_ranges)))
r_error = np.zeros(shape=(xa.shape[1],len(sampling_ranges)))


slice_num=np.zeros(shape=(xa.shape[1]))

#initializing legned array
leg_names=np.empty(len(sampling_ranges), dtype= object)

# this for loop contains all of the fitting steps.  It loops over different amounts sample
# data points.
for div_vars in sampling_ranges:
    #divisor is how many times we'll split up the original data set.
    divisor=5*div_vars
    
    #calculating subset data and angles
    xa_subset = xa[0::(int)(total_points/divisor)]
    ya_subset = ya[0::(int)(total_points/divisor)]
    theta_subset = thetas[0::(int)(total_points/divisor)]
    
    #number of points in the subset
    num_points = len(xa_subset)
    
    n_max=0 # initializing the max number of terms in the Fourier Series
    
    for qq in range(xa.shape[1]):
    #for qq in range(10,12):
        ##defining the max number of  terms in the Fourier Series, i.e. the max frequency
        biggest_division= np.linalg.norm((int)(2*np.pi/np.amax(np.diff(theta_subset[:,qq]))))
        if biggest_division > np.linalg.norm((int)(2*np.pi/((theta_subset[-1,qq]-theta_subset[1,qq]+np.pi)%(2*np.pi)-np.pi))):
            biggest_division = np.linalg.norm((int)(2*np.pi/((theta_subset[-1,qq]-theta_subset[1,qq]+np.pi)%(2*np.pi)-np.pi)))
        if np.mod(biggest_division,2)==0: #num_points is even
            n_max=(biggest_division)/2-1
        elif np.mod(biggest_division,2)==1: #num_points is odd
            n_max=(biggest_division-1)/2
        
        
        n_max=(int)(n_max) # converting to integer
        
        #edge cases of algorithm.  Making sure that the answers don't blow up
        if biggest_division == 2:
            n_max=1
        elif biggest_division ==1:
            n_max = 0
    
        if n_max > 300:
            n_max=300
        #intializing the matrix that contains all of the cos(theta) and sin(theta) terms
        theta_matrix=np.zeros(shape=(num_points,n_max*2+1))
        # n=0 terms
        theta_matrix[:,0]=1
        
        #filling out theta matrix
        for pp in range(num_points):
            for nn in range(1,n_max*2+1):
                if np.mod(nn,2)==1: #cosine terms
                    theta_matrix[pp,nn]=np.cos((nn+1)/2*theta_subset[pp,qq])#np.cos((nn+1)/2*thetas[pp])
                if np.mod(nn,2)==0: #sine terms
                    theta_matrix[pp,nn]=np.sin(nn/2*theta_subset[pp,qq])#np.cos((nn)/2*thetas[pp])
        # code for finding x and y coeffs
        trans_theta_matrix=np.transpose(theta_matrix[:,:])
        inv_theta_matrix=np.linalg.inv(np.matmul(trans_theta_matrix,theta_matrix[:,:]));
        regression_mat=np.matmul(inv_theta_matrix,trans_theta_matrix);
        x_coeffs=np.matmul(regression_mat,xa_subset[:,qq])
        y_coeffs=np.matmul(regression_mat,ya_subset[:,qq])
    
    # Calculation Model points
        theta_step=len(thetas)#100
        theta_model=thetas[:,qq]#np.arange(0,2*np.pi,2*np.pi/theta_step)
        x_model=np.zeros(len(theta_model))
        y_model=np.zeros(len(theta_model))
        theta_model_matrix=np.zeros(shape=(theta_step,n_max*2+1))
        theta_model_matrix[:,0]=1;
        for pp in range(theta_step):
            for nn in range(1,n_max*2+1):
                if np.mod(nn,2)==1: #cosine terms
                    theta_model_matrix[pp,nn]=np.cos((nn+1)/2*theta_model[pp])#np.cos((nn+1)/2*thetas[pp])
                if np.mod(nn,2)==0: #sine terms
                    theta_model_matrix[pp,nn]=np.sin(nn/2*theta_model[pp])#np.cos((nn)/2*thetas[pp])
                
        x_model=np.matmul(theta_model_matrix,x_coeffs)
        y_model=np.matmul(theta_model_matrix,y_coeffs)
        r_model=np.sqrt(x_model**2+y_model**2)
        ra[:,qq] = np.sqrt(xa[:,qq]**2+ya[:,qq]**2)
        
        #calculating errors
        x_error[qq,div_vars-1]= np.sqrt(np.average((x_model-xa[:,qq])**2))
        y_error[qq,div_vars-1]= np.sqrt(np.average((y_model-ya[:,qq])**2))
        r_error[qq,div_vars-1]= np.sqrt(np.average((r_model-ra[:,qq])**2))
        
        slice_num[qq]=qq+1
       
    
#        #optional plotting        
#        print(n_max)
#        print(biggest_division)
#        fig, ax = plt.subplots()
#        
#        ax.plot(xa[:,qq],ya[:,qq],linestyle='none',marker='x',markersize='10',label='Test Data')
#        ax.plot(xa_subset[:,qq],ya_subset[:,qq],linestyle='none',marker='o',markersize='10',label='Subset')
#        ax.plot(x_model,y_model,linestyle='none',marker='o',label='Model')
#        
#        ax.set_xlabel('x (mm)')
#        ax.set_ylabel('y (mm)')
#        ax.legend()
    # legend stuff
    np_str=str(num_points)
    leg_names[div_vars-1]= np_str + " " + "Sample Points"
#%% plotting
fig, ax = plt.subplots()

for zz in range(len(sampling_ranges)):
    ax.plot(slice_num[:],x_error[:,zz],linestyle='none',marker='x',markersize='10',label='Test Data')
    
ax.set_xlabel('Slice Number')
ax.set_ylabel('RMS Error (mm)')
ax.set_title('X Error')
ax.legend(leg_names)

fig, ax = plt.subplots()

for zz in range(len(sampling_ranges)):
    ax.plot(slice_num[:],y_error[:,zz],linestyle='none',marker='x',markersize='10',label='Test Data')
    
ax.set_xlabel('Slice Number')
ax.set_ylabel('RMS Error (mm)')
ax.set_title('Y Error')
ax.legend(leg_names)

fig, ax = plt.subplots()

for zz in range(len(sampling_ranges)):
    ax.plot(slice_num[:],r_error[:,zz],linestyle='none',marker='x',markersize='10',label='Test Data')
    
ax.set_xlabel('Slice Number')
ax.set_ylabel('RMS Error (mm)')
ax.set_title('R Error')
ax.legend(leg_names)



#%% new calculation of points and theta matrix  excluding ~ last 45 degrees
"""
plt.close("all")
total_points = len(xa)
sampling_ranges=np.arange(1,12,1);
#initializing error arrays
x_error = np.zeros(shape=(xa.shape[1],len(sampling_ranges)))
y_error = np.zeros(shape=(xa.shape[1],len(sampling_ranges)))
r_error = np.zeros(shape=(xa.shape[1],len(sampling_ranges)))
slice_num=np.zeros(shape=(xa.shape[1]))
#initializing legned array
leg_names=np.empty(len(sampling_ranges), dtype= object)
for div_vars in sampling_ranges:
    divisor=5*div_vars
    #calculating subset data and angles
    xa_45_deg_chunk=xa[0:790,:]
    ya_45_deg_chunk=ya[0:790,:]
    theta_45_deg_chunk=thetas[0:790,:]
    chunk_points=len(xa_45_deg_chunk)
    xa_subset = xa_45_deg_chunk[0::(int)(chunk_points/divisor)]
    ya_subset = ya_45_deg_chunk[0::(int)(chunk_points/divisor)]
    theta_subset = theta_45_deg_chunk[0::(int)(chunk_points/divisor)]
       
    num_points = len(xa_subset)
    #defining the max number of  terms in the Fourier Series
    
    n_max=0 # initializing the max number of terms in the Fourier Series
    

    for qq in range(xa.shape[1]):
    #for qq in range(10,12):
        #picking max frequency
        biggest_division= np.linalg.norm((int)(2*np.pi/np.amax(np.diff(theta_subset[:,qq]))))
        #test = np.linalg.norm((int)(2*np.pi/((theta_subset[-1,qq]-theta_subset[1,qq]+np.pi)%(2*np.pi)-np.pi)))
        if biggest_division > np.linalg.norm((int)(2*np.pi/((theta_subset[-1,qq]-theta_subset[1,qq]+np.pi)%(2*np.pi)-np.pi))):
            biggest_division = np.linalg.norm((int)(2*np.pi/((theta_subset[-1,qq]-theta_subset[1,qq]+np.pi)%(2*np.pi)-np.pi)))
        if np.mod(biggest_division,2)==0: #num_points is even
            n_max=(biggest_division)/2-1
        elif np.mod(biggest_division,2)==1: #num_points is odd
            n_max=(biggest_division-1)/2
        
        
        n_max=(int)(n_max) # converting to integer
        #edge cases of algorithm
        if biggest_division == 2:
            n_max=1
        elif biggest_division ==1:
            n_max = 0
    
        if n_max > 300:
            n_max=300
        
        
        theta_matrix=np.zeros(shape=(num_points,n_max*2+1))
        theta_matrix[:,0]=1
        for pp in range(num_points):
            for nn in range(1,n_max*2+1):
                if np.mod(nn,2)==1: #cosine terms
                    theta_matrix[pp,nn]=np.cos((nn+1)/2*theta_subset[pp,qq])#np.cos((nn+1)/2*thetas[pp])
                if np.mod(nn,2)==0: #sine terms
                    theta_matrix[pp,nn]=np.sin(nn/2*theta_subset[pp,qq])#np.cos((nn)/2*thetas[pp])
    
    #for qq in range(xa.shape[1]):
    #for qq in range(10,12):
        trans_theta_matrix=np.transpose(theta_matrix[:,:])
        inv_theta_matrix=np.linalg.inv(np.matmul(trans_theta_matrix,theta_matrix[:,:]));
        regression_mat=np.matmul(inv_theta_matrix,trans_theta_matrix);
        x_coeffs=np.matmul(regression_mat,xa_subset[:,qq])
        y_coeffs=np.matmul(regression_mat,ya_subset[:,qq])
    
    # Calculation Model points
        theta_step=len(thetas)#100
        theta_model=thetas[:,qq]#np.arange(0,2*np.pi,2*np.pi/theta_step)
        x_model=np.zeros(len(theta_model))
        y_model=np.zeros(len(theta_model))
        theta_model_matrix=np.zeros(shape=(theta_step,n_max*2+1))
        theta_model_matrix[:,0]=1;
        for pp in range(theta_step):
            for nn in range(1,n_max*2+1):
                if np.mod(nn,2)==1: #cosine terms
                    theta_model_matrix[pp,nn]=np.cos((nn+1)/2*theta_model[pp])#np.cos((nn+1)/2*thetas[pp])
                if np.mod(nn,2)==0: #sine terms
                    theta_model_matrix[pp,nn]=np.sin(nn/2*theta_model[pp])#np.cos((nn)/2*thetas[pp])
                
        x_model=np.matmul(theta_model_matrix,x_coeffs)
        y_model=np.matmul(theta_model_matrix,y_coeffs)
        r_model=np.sqrt(x_model**2+y_model**2)
        ra[:,qq] = np.sqrt(xa[:,qq]**2+ya[:,qq]**2)
        
        
        x_error[qq,div_vars-1]= np.sqrt(np.average((x_model-xa[:,qq])**2))
        y_error[qq,div_vars-1]= np.sqrt(np.average((y_model-ya[:,qq])**2))
        r_error[qq,div_vars-1]= np.sqrt(np.average((r_model-ra[:,qq])**2))
        slice_num[qq]=qq+1
       
    
        
#        print(n_max)
#        print(biggest_division)
#        fig, ax = plt.subplots()
#        
#        ax.plot(xa[:,qq],ya[:,qq],linestyle='none',marker='x',markersize='10',label='Test Data')
#        ax.plot(xa_subset[:,qq],ya_subset[:,qq],linestyle='none',marker='o',markersize='10',label='Subset')
#        ax.plot(x_model,y_model,linestyle='none',marker='o',label='Model')
#        
#        ax.set_xlabel('x (mm)')
#        ax.set_ylabel('y (mm)')
#        ax.legend()
    
    np_str=str(num_points)
    leg_names[div_vars-1]= np_str + " " + "Sample Points"
    #%%
fig, ax = plt.subplots()

for zz in range(len(sampling_ranges)):
    ax.plot(slice_num[:],x_error[:,zz],linestyle='none',marker='x',markersize='10',label='Test Data')
    
ax.set_xlabel('Slice Number')
ax.set_ylabel('RMS Error (mm)')
ax.set_title('X Error')
ax.legend(leg_names)

fig, ax = plt.subplots()

for zz in range(len(sampling_ranges)):
    ax.plot(slice_num[:],y_error[:,zz],linestyle='none',marker='x',markersize='10',label='Test Data')
    
ax.set_xlabel('Slice Number')
ax.set_ylabel('RMS Error (mm)')
ax.set_title('Y Error')
ax.legend(leg_names)

fig, ax = plt.subplots()

for zz in range(len(sampling_ranges)):
    ax.plot(slice_num[:],r_error[:,zz],linestyle='none',marker='x',markersize='10',label='Test Data')
    
ax.set_xlabel('Slice Number')
ax.set_ylabel('RMS Error (mm)')
ax.set_title('R Error')
ax.legend(leg_names)
"""