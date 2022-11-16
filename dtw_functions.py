# -*- coding: utf-8 -*-
"""
Functions for pre-processing lithological data, followed by clustering using Dynamic Time Warping algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

from ipywidgets import interact

from matplotlib.colors import LogNorm   

import numpy.ma as ma

#Class for analyzing and visualizing the data
class Data:
    
    #Initialize class using input dataframe
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    #Display rows from row_start to row_end
    def display(self, row_start, row_end):
        df = self.dataframe
        print(df[row_start:row_end])
      
    #Sort a column of the dataframe
    def sort_variable(self, variable):
        df = self.dataframe
        variable_sort = np.sort(np.unique(df[variable]))           
        return variable_sort

    #Determine minimum of a column of the dataframe
    def find_min(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_min = np.min(variable_sort)
        return variable_min
    
    #Determine maximum of a column of the dataframe
    def find_max(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_max = np.max(variable_sort)
        return variable_max
  
    #Determine step size of a variable in a column
    def find_step(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_step = variable_sort[1] - variable_sort[0]
        return variable_step
        
    #Find number of unique values of a variable in the dataframe
    def find_num_datapoints(self, variable):
        variable_sort = self.sort_variable(variable)
        num_datapoints = variable_sort.shape[0]
        return num_datapoints
    
    #Find mean of a variable in the dataframe
    def find_mean(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_mean = np.mean(variable_sort)
        return variable_mean       
     
    #Find standard deviation of a variable in the dataframe
    def find_std(self, variable):
        variable_sort = self.sort_variable(variable)
        variable_std = np.std(variable_sort)
        return variable_std
     
    #Summary of data
    def data_summary(self):
        
        summary = {}
        summary['variable'] = ['x', 'y', 'z']
        summary['num_values'] = [self.find_num_datapoints('x'), self.find_num_datapoints('y'), self.find_num_datapoints('z')]
        summary['min'] = [self.find_min('x'), self.find_min('y'), self.find_min('z')]
        summary['max'] = [self.find_max('x'), self.find_max('y'), self.find_max('z')]
        summary['step_size'] = [self.find_step('x'), self.find_step('y'), self.find_step('z')] 
        summary['mean'] = [self.find_mean('x'), self.find_mean('y'), self.find_mean('z')]
        summary['std'] = [self.find_std('x'), self.find_std('y'), self.find_std('z')]
        
        df_summary = pd.DataFrame(data=summary)
        print(df_summary.round(2))

             
    #Display cross-section along coordinate variable
    def display_cross_section(self, variable=None, value=None, quantity=None, logscale=True, vmin=1, vmax=1e3, xticklabels=10, yticklabels=20, cmap=None):

        df = self.dataframe
        
        N_x = self.find_num_datapoints('x')
        N_y = self.find_num_datapoints('y')
        N_z = self.find_num_datapoints('z')
        
        def f(value):
            
            slice = df[(df[variable]==value)]

            if variable=='z':
                slice = slice.pivot('y','x',quantity)
                aspect = N_x/N_y
            if variable=='x':
                slice = slice.pivot('z','y',quantity)
                aspect = N_y/N_z
            if variable=='y':
                slice = slice.pivot('z','x',quantity)
                aspect = N_x/N_z                      
            if logscale:    
                ax = sns.heatmap(slice, norm=LogNorm(vmin=vmin, vmax=vmax), xticklabels=xticklabels, yticklabels=yticklabels, cmap=cmap)
            else:
                ax = sns.heatmap(slice, vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=yticklabels, cmap=cmap)
                
            ax.set_aspect(aspect)
            ax.invert_yaxis()
        
        #Display interactive cross-section
        if value is None:         
            
            min_value = self.find_min(variable)
            max_value = self.find_max(variable)
            step = self.find_step(variable)
                
            interact(f, value = (min_value, max_value, step))
            
        #Display one slice    
        else:
            f(value)
            
    #Convert dataframe to numpy array
    def dataframe_to_numpy_array(self):
        df = self.dataframe
        data = df.to_numpy()
        return data    
    
    #Mask numpy array
    def mask_data(self, data, mask_value=-9999.0):
        masked_data = ma.masked_values(data, mask_value)
        return masked_data
    
    #Normalize numpy array
    def normalize_data(self, data):
        self.data_mean = np.mean(data, axis=0)
        self.data_std = np.std(data, axis=0)
        normalized_data = (data - self.data_mean) / self.data_std
        self.normalized_data = normalized_data
        return normalized_data        

def stack_xy_probs(dataframe, N_x, N_y, N_z):

    dimension_cube = [N_x, N_y, N_z]
    num_rows_dataframe = len(dataframe)    
    indices_dataframe = np.arange(0, num_rows_dataframe)
    
    indices_unraveled = np.unravel_index(indices_dataframe, dimension_cube)
    
    num_variables = len(dataframe.columns)    
        
    data_cube = np.zeros((N_x,N_y,N_z,num_variables))
  
    for i in range(num_variables):
        data_i = np.zeros((N_x, N_y, N_z))
        data_i[indices_unraveled[0], indices_unraveled[1], indices_unraveled[2]] = dataframe[dataframe.columns[i]]
        data_cube[:,:,:,i] = data_i
        
    #List of data stacks
    data_stacks = []
    #x-position indices of data stacks
    data_stacks_nx = []
    #y-position indices of data stacks
    data_stacks_ny = []    
    
    for ix in range(N_x):
        for iy in range(N_y):
            data_stack_z = data_cube[ix,iy].tolist()
            data_stack = [item for item in data_stack_z if item != [-1,-1,-1,-1] and item!=[0,0,0,0]]
            if data_stack:
                if len(data_stack)==1:
                    data_stack = [data_stack[0], data_stack[0]]
                data_stacks.append(data_stack)
                data_stacks_nx.append(ix)
                data_stacks_ny.append(iy)
    return data_stacks, data_stacks_nx, data_stacks_ny
