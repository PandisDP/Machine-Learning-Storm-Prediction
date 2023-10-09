import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as pltcolors
from scipy.stats import kurtosis, skew
import os
from scipy import stats
class data_set(object):
    def __init__(self):
        pass
    def prepare_data_to_ml(self,df,params):
        df = df.set_index(params[0])
        df=df.reset_index(drop= True)
        df = df.drop([params[1]], axis=1)
        df=df.sample(frac=1)
        return df

    def correlations_c(self,dfx,not_use):
        print("Matriz de Correlaciones")
        if len(not_use) != 0:
            df_new= dfx.drop(not_use, axis=1)
            corr = df_new.corr()
        else:
            df_new=dfx
            corr = dfx.corr()
        return corr,df_new
    
    def correlations(self, df, not_use,PATH_OUTPUT,namex="Demo"):
        if len(not_use) != 0:
            df_new = df.drop(not_use, axis=1)
            numeric_columns = df_new.select_dtypes(include=['number']).columns
            corr = df_new[numeric_columns].corr()
        else:
            df_new = df
            numeric_columns = df.select_dtypes(include=['number']).columns
            corr = df[numeric_columns].corr()
        styled_corr =corr.style.background_gradient()
        #Save corr
        plt.figure(figsize=(10, 6))
        plt.imshow(styled_corr.data, aspect='auto', cmap='RdYlBu_r')
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
        plt.yticks(range(len(corr.index)), corr.index)
        plt.colorbar()
        plt.title("Correlations")
        ruta_destino = os.path.join(os.getcwd(),PATH_OUTPUT, f"Corr_{namex}.png")
        plt.savefig(ruta_destino)
        plt.close() 
   
    def outliers_clean_base(self,df,target_list):
        df_temp= df.copy()
        df_over= self.__Outlier_Analysis(df_temp,target_list)
        return df_over
    def drop_high_null_columns(self,df, threshold=0.8):
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be a float between 0 and 1.")
        to_drop=[]
        for col in df.columns:
            null_ = df[col].isnull().sum()
            if null_/df.shape[0]>threshold:
                to_drop.append(col)
        df_cleaned = df.drop(columns=to_drop)
        # Rellenar los NaN con la media de su columna respectiva
        df_filled = df_cleaned.apply(lambda col: col.fillna(col.mean()), axis=0)
        return df_filled

    def balance_subsample(self,df,nums):
        count_clase=df[nums].value_counts()
        count_clase=count_clase.sort_values(ascending = False)
        df_over=df[df[nums]==count_clase.index[0]]
        for i in range(0,count_clase.size):
            if i>0:
                df_temp= df[df[nums]==count_clase.index[i]]
                df_temp_under= df_temp.sample(count_clase[count_clase.index[0]],replace=True)
                df_over= pd.concat([df_temp_under,df_over],axis=0)      
        return df_over

    def __Outlier_Analysis(self,data,target):
        for xena in data.columns:
            if xena in target:
                Q1 = data[xena].quantile(0.25)
                Q3 = data[xena].quantile(0.75)
                IQR = Q3 - Q1
                LR = Q1 -(1.5 * IQR)
                UR = Q3 + (1.5 * IQR)
                data.drop(data[(data[xena] > UR) | (data[xena]< LR)].index, inplace=True)
        return data

    def __Analysis_Indep(self,obj,namex,target,path):
        df_temp = obj[obj[target]== namex].copy()
        df_temp=df_temp.drop(columns= [target])
        for i_elm in df_temp.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_temp[i_elm],stat="density", common_norm=False)
            plt.title(f"{namex} - {i_elm}") 
            ruta_destino = os.path.join(os.getcwd(),path, f"_{namex}_{i_elm}.png")
            plt.savefig(ruta_destino)
            plt.close()
   
    def __Analysis_byAtrr(self, obj, atr, lst_p,target,path):
        plt.figure(figsize=(10, 6))
        for i in lst_p:
            f_temp = obj[obj[target] == i].copy()
            sns.histplot(data=f_temp[atr])
        ruta_destino = os.path.join(os.getcwd(),path, f"{atr}_{i}.png")
        plt.title(atr)
        plt.legend(labels=lst_p)
        plt.savefig(ruta_destino)
        plt.close()  # Cierra el gráfico actual antes de crear el siguiente

    def multi_picture(self,df,lst_p,lst_a,target,path):
        for i in lst_p:
            print("Analysis of {}".format(i))
            self.__Analysis_Indep(df,i,target,path)
        for j in lst_a:
            print("Analysis of {}".format(j))
            self.__Analysis_byAtrr(df, j, lst_p,target,path)

    def load_database(self,rut_file):
        df = pd.read_csv(rut_file)
        return df