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

from ase import Atoms
from ase.io import read, write
from ase.visualize import view

class Model_importer:
    def __init__(self,data_path,model_save_path):#descriptor_dict,position_count,dataframe):
        self.data_path = data_path #str: stored data path/filename
        self.model_save_path = model_save_path #str: stored model path

    def Import_model(self):
        model = Sequential([
                            Dense(64,input_shape=(52,),activation='relu', use_bias=False),
                            Dense(64,activation='relu',use_bias=False),
                            Dense(12,activation='relu',use_bias=False),
                            Dense(1,activation="linear",use_bias=False),])
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss='mse', metrics=['mae','mse'])
        model.load_weights(self.model_save_path)
        return model
    
    def Import_data(self):
        with open(self.data_path, 'rb') as f:
            X_train = np.load(f)
            X_val = np.load(f)
            X_test = np.load(f)
            Y_train = np.load(f)
            Y_val = np.load(f)
            Y_test = np.load(f)
        return X_train,X_val,X_test,Y_train,Y_val,Y_test
    
    def Model_performance(self,model,X_train,X_test,Y_train,Y_test ,save_img=False):
        yhat_train =list( model.predict(X_train)[:,0] )
        yhat_test  =list( model.predict(X_test)[:,0]  ) 
        preds=yhat_test+yhat_train

        #Real Values
        y_test= list(Y_test[:,0])
        y_train=list(Y_train[:,0])
        real = y_test+y_train

        labels=[]
        for i in range(len(yhat_test)):
            labels.append('test')
        for i in range(len(yhat_train)):
            labels.append('train')

        d = {'Predicted': preds, 'Real': real,'Label':labels}
        df_scatter = pd.DataFrame(data=d)

        #MAE Values
        print(f'MAE train:  {str(mean_absolute_error(y_train, yhat_train))[:6]} eV')
        print(f'MAE test :  {str(mean_absolute_error(y_test, yhat_test))[:6]} eV')

        #Calculated std dev
        std_dev = np.std ( df_scatter['Predicted'] - df_scatter['Real'] )

        x = [-8,4]
        y = [-8-std_dev,4-std_dev]
        hue=['low','low']
        df1 = pd.DataFrame(data={'x':x,'lines': y,'hue':hue})

        x = [-8,4]
        y = [-8+std_dev,4+std_dev]
        hue=['up','up']
        df2 = pd.DataFrame(data={'x':x,'lines': y,'hue':hue})

        #Plot using seaborn scatterplot
        x4=sns.scatterplot(data=df_scatter, x='Predicted',y='Real',hue='Label',marker='o', color='b')
        x4.set(xlim=(-5.5,-1))
        x4.set(ylim=(-5.5,-1))
        x4.plot([-8,4],[-8,4],color='black',alpha=0.5) 
        x4=sns.lineplot(x='x',y='lines',data=df1,color='black',alpha  = 0.5)
        x4.lines[1].set_linestyle("--")
        x4=sns.lineplot(x='x',y='lines',data=df2,color='black',alpha  = 0.5)
        x4.lines[2].set_linestyle("--")
        x4.set_xlabel("Predicted Eads (eV)", fontsize = 15)
        x4.set_ylabel("DFT Eads (eV)", fontsize = 15)
        x4.set_title("Parity Plot (with 1 Standard Deviation Range)", fontsize = 20)
        plt.tight_layout()
        if save_img:
            plt.savefig('NN_Parity_Plot')
        #--------------------------------
        return mean_absolute_error(y_test, yhat_test)
    
class Strucutre_Generator():
    def __init__(self,dictionnary,model):#descriptor_dict,position_count,dataframe):
        self.dictionnary = dictionnary
        self.model = model

    def Convert_line(self, line):
        label_holders=[]
        for i in range(52):
            label_holders.append(str(i))
        readable_input = []
        for i in line:
            readable_input += self.dictionnary[i]

        df_2D = pd.DataFrame(data=np.reshape(np.array(readable_input),(1,52)),columns=label_holders)
        return df_2D
        
        
    def Random_datapoint(self, ele): #generate random datapoint in ML predictable format
        template = ['Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu']
        replacements=[]
        for k in range(13):
            x = (random.randrange(0,4,1)) #value from 0 to 3
            if x == 0: #if we hit the 25% chance to switch element
                replacements.append(k) 
        for i in replacements:
            template[i] = ele
        converted=self.Convert_line(template)

        return template,converted
        
    def Generate_structures_2n(self,pred_ele,count=100,save=False):
        t1=time.time()        
        predictions_str = []
        predictions=[]
        struct_pred = []
        print(f'Generating {pred_ele} BACs...')        
        for i in range(count):
            random_template,datapoint = self.Random_datapoint(pred_ele)
            prediction = self.model.predict(datapoint)

            predictions_str.append(str(prediction[0][0]))
            predictions.append(prediction[0][0])
            struct_pred.append(random_template)
        t2=time.time()      
        print(f'2n Structure Gen. Runtime (s): {t2-t1}')

        plt.hist(predictions,bins=25)
        plt.title(f'Structure predictions with {pred_ele} binary alloy')
        plt.xlabel('Eads (eV)')
        plt.ylabel('Frequency')
        
        if save:
            json_structures = f"structures_{pred_ele}.json"
            with open(json_structures, 'w') as f:
                json.dump(struct_pred, f)
                
            json_Eads = f"Eads_{pred_ele}.json"
            with open(json_Eads, 'w') as f:
                json.dump(predictions_str, f)
                
        self.predictions_2n = predictions
        self.predictions_structures_2n = struct_pred
        self.alloyed_element_2n = pred_ele
        
        return predictions_str,predictions,struct_pred
    
    def Get_generation_stats(self,preds):
        Mean = np.mean(preds)
        print(f'Mean: {str(Mean)[:5]}')
        Range = np.max(preds) - np.min(preds)
        print (f'Range: {str(Range)[:5]}')
        best_ads = np.max(preds)
        print ('Best Ads: '+str(best_ads)[:5])
        
    def Get_optimal_structure(self,preds,structs,elements):        
        for i in range(len(preds)):
            if preds[i] == np.min(preds):
                idx=i
                
        replacement_idx=[]
        for i in range(len(structs[idx])):
            if structs[idx][i] in elements:
                replacement_idx.append(i+1)
                
        return structs[idx],replacement_idx
    
    def New_design_2n(self,replacements,element,slab=None,save=False,formatting='espresso-in'):
        if slab is None:
            sample_atom = read('Cu_Pure',format='vasp')
        if slab is not None:
            sample_atom = slab
        cus=[]
        
        for i in range(48):
            if i == 24: #atom 1
                if 1 in replacements:
                    cus.append(element)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)
                else:
                    cus.append('Cu')

            elif i == 25: #atom 2
                if 2 in replacements:
                    cus.append(element)
                    #print(i)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(element)            
                else:
                    cus.append('Cu')

            elif i == 26: #atom 3
                if 3 in replacements:
                    cus.append(element)
                    #print(i)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)                            
                else:
                    cus.append('Cu')

            elif i == 28: #atom 4
                if 4 in replacements:
                    cus.append(element) 
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                else:
                    cus.append('Cu')

            elif i == 29: #atom 5
                if 5 in replacements:
                    cus.append(element) 
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                else:
                    cus.append('Cu')

            elif i == 30: #atom 6
                if 6 in replacements:
                    cus.append(element)   
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                else:
                    cus.append('Cu')

            elif i == 32: #atom 7
                if 7 in replacements:
                    cus.append(element)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)                    
                else:
                    cus.append('Cu')

            elif i == 33: #atom 8
                if 8 in replacements:
                    cus.append(element)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)                 
                else:
                    cus.append('Cu')

            elif i == 34: #atom 9
                if 9 in replacements:
                    cus.append(element)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                else:
                    cus.append('Cu')

            elif i == 37: #atom 10
                if 10 in replacements:
                    cus.append(element)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                    #print(i)
                else:
                    cus.append('Cu')

            elif i == 38: #atom 11
                if 11 in replacements:
                    cus.append(element)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                    #print(i)
                else:
                    cus.append('Cu')

            elif i == 41: #atom 12
                if 12 in replacements:
                    cus.append(element) 
                    #print(i)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                else:
                    cus.append('Cu')

            elif i == 42: #atom 13
                if 13 in replacements:
                    cus.append(element) 
                    #print(i)
                elif sample_atom[i].symbol != 'Cu':
                    cus.append(sample_atom[i].symbol)            
                else:
                    cus.append('Cu')

            else:
                cus.append('Cu')
        sample_atom.set_chemical_symbols(cus)
        if save:
            file_name = f'{element}.relax'
            write(file_name,atm2,format=formatting )            
        return sample_atom
        
    def Random_datapoint_3n(self, ele1,ele2):
        template = ['Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu','Cu']
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
            template[i] = ele1
        for i in ele2_replace:
            template[i] = ele2
        
        converted=self.Convert_line(np.array(template))
        return template,converted
    
    def Generate_structures_3n(self,e1,e2,count=100,save=False):
        t1=time.time()

        predictions_str=[]
        predictions = []
        struct_pred = [] #list of lists


        #for j in range(len(ele_combinations)):
            #e1=ele_combinations[j][0]
            #e2=ele_combinations[j][1]
            #e1_e2= e1+'_'+e2
        print(f'Generating {e1}/{e2} TACs...')
        for i in range(count):
            template,datapoint = self.Random_datapoint_3n(e1,e2)

            predictions_str.append(str(self.model.predict(datapoint)[0][0]))
            predictions.append(self.model.predict(datapoint)[0][0])
            struct_pred.append(template)

                #ele_combo.append(e1_e2)
        t2=time.time()
        print(f'3n Structure Gen. Runtime (s): {t2-t1}')

        if save:
            json_structures = f"structures_{e1}_{e2}.json"
            with open(json_structures, 'w') as f:
                json.dump(struct_pred, f)

            json_Eads = f"Eads_{e1}_{e2}.json"
            with open(json_Eads, 'w') as f:
                json.dump(predictions_str, f)

        self.predictions_3n = predictions
        self.predictions_structures_3n = struct_pred
        self.alloyed_element_3n = e1+e2
        return predictions_str,predictions,struct_pred        

    def Symbol_to_index(self,ele,structure): #ele=str, structure= list
        idx=[]
        for i in range(len(structure)):
            if structure[i] == ele:
                idx.append(i+1)
            else:
                p=0
        return idx        
         
    def New_design_3n(self,e1,e2,structure,save=False,formatting='espresso-in'):
        rep1 = self.Symbol_to_index(e1,structure)
        rep2 = self.Symbol_to_index(e2,structure)
        
        atm1 = self.New_design_2n(rep1,e1)
        atm2 = self.New_design_2n(rep2,e2,atm1)
        
        if save:
            file_name = f'{e1}_{e2}.relax'
            write(file_name,atm2,format=formatting )            
        return atm1,atm2                            
        
        
        
        
        
        
        
        
        
        
        
        