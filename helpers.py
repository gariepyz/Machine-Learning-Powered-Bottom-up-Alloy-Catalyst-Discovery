#!/usr/bin/env python
# coding: utf-8

#package imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import json
import itertools
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

#Convert df line of elements into ML readable format

def convert_line(line):
    #line is row of dataframe.iloc to convert
    #line = df_X.iloc[0]
    s0 = [1,0,0,0,0,0,0,0,0,0,0,0,0]
    s1 = [0,1,0,0,0,0,0,0,0,0,0,0,0]
    s2 = [0,0,1,0,0,0,0,0,0,0,0,0,0]
    s3 = [0,0,0,1,0,0,0,0,0,0,0,0,0]
    s4 = [0,0,0,0,1,0,0,0,0,0,0,0,0]
    s5 = [0,0,0,0,0,1,0,0,0,0,0,0,0]
    s6 = [0,0,0,0,0,0,1,0,0,0,0,0,0]
    s7 = [0,0,0,0,0,0,0,1,0,0,0,0,0]
    s8 = [0,0,0,0,0,0,0,0,1,0,0,0,0]
    s9 = [0,0,0,0,0,0,0,0,0,1,0,0,0]
    s10 =[0,0,0,0,0,0,0,0,0,0,1,0,0]
    s11 =[0,0,0,0,0,0,0,0,0,0,0,1,0]
    s12 =[0,0,0,0,0,0,0,0,0,0,0,0,1]
    dorbitals = {
         'Cu':[4,11,1.90,0],
         'Au':[6,11,2.54,0],
         'Ag':[5,11,1.93,0],
         'Pt':[6,10,2.28,1],
         'Pd':[5,10,2.20,0],
         'Ir':[6,9,2.20,3],
         'Os':[6,8,2.20,4],
         'Ru':[5,8,2.20,3],
         'Rh':[5,9,2.28,2]} 

    df_np = np.empty((1,13,17))
    for j in range(df_np.shape[0]):
        l=np.empty((13,17))
        row = line
        for i in range(13):
            if i==0:
                gs = dorbitals[row[i]]
                g = gs+s0
                l[i,:] = g
            if i==1:
                gs = dorbitals[row[i]]
                g = gs+s1
                l[i,:] = g
            if i==2:
                gs = dorbitals[row[i]]
                g = gs+s2
                l[i,:] = g            
            if i==3:
                gs = dorbitals[row[i]]
                g = gs+s3
                l[i,:] = g
            if i==4:
                gs = dorbitals[row[i]]
                g = gs+s4
                l[i,:] = g
            if i==5:
                gs = dorbitals[row[i]]
                g = gs+s5
                l[i,:] = g
            if i==6:
                gs = dorbitals[row[i]]
                g = gs+s6
                l[i,:] = g
            if i==7:
                gs = dorbitals[row[i]]
                g = gs+s7
                l[i,:] = g
            if i==8:
                gs = dorbitals[row[i]]
                g = gs+s8
                l[i,:] = g
            if i==9:
                gs = dorbitals[row[i]]
                g = gs+s9
                l[i,:] = g

            if i==10:
                gs = dorbitals[row[i]]
                g = gs+s10
                l[i,:] = g
            if i==11:
                gs = dorbitals[row[i]]
                g = gs+s11
                l[i,:] = g
            if i==12:
                gs = dorbitals[row[i]]
                g = gs+s12
                l[i,:] = g            
        df_np[j,:,:] = l
    df_2D = np.zeros((1,13*4))
    point = df_np[0] # 13x17 dataset
    l = []
    for j in range(point.shape[0]): #iterate each row of datapoint
        line1 = point[j,0:4]
        for k in line1:
            l.append(k)
    df_2D[0,:] = l

    descripts = [
        'group_1','period_1','EN_1','Nied_1',
        'group_2','period_2','EN_2','Nied_2',
        'group_3','period_3','EN_3','Nied_3',
        'group_4','period_4','EN_4','Nied_4',
        'group_5','period_5','EN_5','Nied_5',
        'group_6','period_6','EN_6','Nied_6',
        'group_7','period_7','EN_7','Nied_7',
        'group_8','period_8','EN_8','Nied_8',
       'group_9','period_9','EN_9','Nied_9',
       'group_10','period_10','EN_10','Nied_10',
        'group_11','period_11','EN_11','Nied_11',
        'group_12','period_12','EN_12','Nied_12',
        'group_13','period_13','EN_13','Nied_13']

    df_2D = pd.DataFrame(data=df_2D,
                     columns=descripts)        
    return df_2D #df of 1x52

#generate random datapoint in ML predictable format
def random_multi_ele_datapoint(ele1,ele2):
    s0 = [1,0,0,0,0,0,0,0,0,0,0,0,0]
    s1 = [0,1,0,0,0,0,0,0,0,0,0,0,0]
    s2 = [0,0,1,0,0,0,0,0,0,0,0,0,0]
    s3 = [0,0,0,1,0,0,0,0,0,0,0,0,0]
    s4 = [0,0,0,0,1,0,0,0,0,0,0,0,0]
    s5 = [0,0,0,0,0,1,0,0,0,0,0,0,0]
    s6 = [0,0,0,0,0,0,1,0,0,0,0,0,0]
    s7 = [0,0,0,0,0,0,0,1,0,0,0,0,0]
    s8 = [0,0,0,0,0,0,0,0,1,0,0,0,0]
    s9 = [0,0,0,0,0,0,0,0,0,1,0,0,0]
    s10 =[0,0,0,0,0,0,0,0,0,0,1,0,0]
    s11 =[0,0,0,0,0,0,0,0,0,0,0,1,0]
    s12 =[0,0,0,0,0,0,0,0,0,0,0,0,1]
    dorbitals = {
         'Cu':[4,11,1.90,0],
         'Au':[6,11,2.54,0],
         'Ag':[5,11,1.93,0],
         'Pt':[6,10,2.28,1],
         'Pd':[5,10,2.20,0],
         'Ir':[6,9,2.20,3],
         'Os':[6,8,2.20,4],
         'Ru':[5,8,2.20,3],
         'Rh':[5,9,2.28,2]} 
    #print(dorbitals)
    template1 = ['Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu']
    replacements=[]
    for k in range(13):
        x = (random.randrange(0,3,1)) #value from 0 to 3
        if x == 0: #if we hit the 25% chance to switch element
            replacements.append(k) #index of atoms to be replaced
    
    ele1_replace=[]
    ele2_replace=[]
    for j in replacements:
        x = (random.randrange(0,2,1)) #value of 0 or 1
        if x == 0: #if we hit the 50% chance to switch element 1
            ele1_replace.append(j) #index of atoms to be replaced
        if x == 1: #if we hit the 50% chance to switch element 1
            ele2_replace.append(j) #index of atoms to be replaced
    
    for i in ele1_replace:
        template1[i] = ele1
    for i in ele2_replace:
        template1[i] = ele2
        
    #print(np.array(template1))
    converted=convert_line(np.array(template1))
    return template1,converted

def get_spectrum(ele,df):
    #ele = 'Ru_Rh'
    df1 = df[df['Elements']==ele]
    Mean = np.mean(df1['Eads pred (eV)'])
    print ('Mean: '+str(Mean)[:5])
    Range = np.max(df1['Eads pred (eV)']) - np.min(df1['Eads pred (eV)'])
    print ('Range: '+str(Range)[:5])
    return df1


def new_design(replacements,ele,atoms=None):
    if atoms is None:
        x = read('Cu_Pure',format='vasp')
    if atoms is not None:
        x = atoms
    cus=[]
    
    for i in range(48):
        if i == 24: #atom 1
            if 1 in replacements:
                cus.append(ele)
                #print(i)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)
            else:
                cus.append('Cu')
                
        elif i == 25: #atom 2
            if 2 in replacements:
                cus.append(ele)
                #print(i)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
            else:
                cus.append('Cu')
                
        elif i == 26: #atom 3
            if 3 in replacements:
                cus.append(ele)
                #print(i)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)                            
            else:
                cus.append('Cu')
                
        elif i == 28: #atom 4
            if 4 in replacements:
                cus.append(ele) 
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
            else:
                cus.append('Cu')
                
        elif i == 29: #atom 5
            if 5 in replacements:
                cus.append(ele) 
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
            else:
                cus.append('Cu')
                
        elif i == 30: #atom 6
            if 6 in replacements:
                cus.append(ele)   
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
            else:
                cus.append('Cu')
                
        elif i == 32: #atom 7
            if 7 in replacements:
                cus.append(ele)
            else:
                cus.append('Cu')
                
        elif i == 33: #atom 8
            if 8 in replacements:
                cus.append(ele) 
            else:
                cus.append('Cu')
                
        elif i == 34: #atom 9
            if 9 in replacements:
                cus.append(ele)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
            else:
                cus.append('Cu')
                
        elif i == 37: #atom 10
            if 10 in replacements:
                cus.append(ele)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 38: #atom 11
            if 11 in replacements:
                cus.append(ele)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 41: #atom 12
            if 12 in replacements:
                cus.append(ele) 
                #print(i)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
            else:
                cus.append('Cu')
                
        elif i == 42: #atom 13
            if 13 in replacements:
                cus.append(ele) 
                #print(i)
            elif x[i].symbol != 'Cu':
                cus.append(x[i].symbol)            
            else:
                cus.append('Cu')
                
        else:
            cus.append('Cu')
            #print(i)
    #print(len(cus))
    x.set_chemical_symbols(cus)
    #view(x)
    return x


def Symbol_to_index(ele,structure): #ele=str, structure= list
    idx=[]
    for i in range(len(structure)):
        if structure[i] == ele:
            idx.append(i+1)
        else:
            p=0
    return idx

def random_datapoint(ele): #generate random datapoint in ML predictable format
    template1 = ['Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu']
    replacements=[]
    for k in range(13):
        x = (random.randrange(0,4,1)) #value from 0 to 3
        if x == 0: #if we hit the 25% chance to switch element
            replacements.append(k) 
    #print(replacements)
    for i in replacements:
        template1[i] = ele
    #print(np.array(template1))
    converted=convert_line(np.array(template1))
    return template1,converted
    
def new_design_BAC(replacements,ele):
    x = read('CONTCAR_vasp.vasp')
    cus=[]
    
    for i in range(52):
        if i == 24: #atom 1
            if 1 in replacements:
                cus.append(ele)
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 46: #atom 2
            if 2 in replacements:
                cus.append(ele)
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 47: #atom 3
            if 3 in replacements:
                cus.append(ele)
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 26: #atom 4
            if 4 in replacements:
                cus.append(ele) 
            else:
                cus.append('Cu')
                
        elif i == 27: #atom 5
            if 5 in replacements:
                cus.append(ele)  
            else:
                cus.append('Cu')
                
        elif i == 28: #atom 6
            if 6 in replacements:
                cus.append(ele)   
            else:
                cus.append('Cu')
                
        elif i == 30: #atom 7
            if 7 in replacements:
                cus.append(ele)
            else:
                cus.append('Cu')
                
        elif i == 31: #atom 8
            if 8 in replacements:
                cus.append(ele) 
            else:
                cus.append('Cu')
                
        elif i == 48: #atom 9
            if 9 in replacements:
                cus.append(ele)
            else:
                cus.append('Cu')
                
        elif i == 39: #atom 10
            if 10 in replacements:
                cus.append(ele)
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 40: #atom 11
            if 11 in replacements:
                cus.append(ele)
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 49: #atom 12
            if 12 in replacements:
                cus.append(ele) 
                #print(i)
            else:
                cus.append('Cu')
                
        elif i == 50: #atom 13
            if 13 in replacements:
                cus.append(ele) 
                #print(i)
            else:
                cus.append('Cu')
                
        else:
            cus.append('Cu')
            #print(i)
    #print(len(cus))
    x.set_chemical_symbols(cus)
    view(x)
    return x
