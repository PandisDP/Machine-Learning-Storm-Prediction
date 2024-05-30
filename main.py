import yaml
from NeuronalNetwork import RedNeuronal
from dataset import data_set
from database_managment import dataset_managment
import pandas as pd

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def DataEngineering(PATH,PATH_OUTPUT,drop_columns=[],target_obj='KP'):
    print("Data Engineering Process")
    db=data_set()
    df=db.load_database(PATH,drop_columns)
    # Clean NANclear
    print("Drop Null Columns  of 5% of data")
    df=db.drop_high_null_columns(df,0.05,target_obj)
    # Data Analysis of Correlations
    print("Correlatios of Data")
    db.correlations(df,[],PATH_OUTPUT,"Initial")
    # Clear Ouliers
    print("Outliers Clean of Data")
    list_cols=df.columns.to_list()[:-1]
    print(list_cols)
    df_balance=db.outliers_clean_base(df,list_cols)
    print('Balance of Data')
    # Balance of data con Sub Sumple Method
    df_balance =db.balance_subsample(df_balance,target_obj)
    print('Statistics of Data')
    # Estadisticas y correlaciones
    db.multi_picture(df_balance,[0,50,60,70,80],df_balance.columns.to_list(),target_obj,PATH_OUTPUT) 
    return df_balance
def Training_Process(df,target_obj,PATH_MODELS):
    print("Training Process")
    # Create Model of Training and Testing with Red Neuronal
    rn = RedNeuronal(df,target_obj,42)
    #This method generate the best hyperparameters of Neuronal Network like activation functiona and neurones of each level
    #print("Hyperparameters of Neuronal Network")
    #params_hyper= rn.find_best_Hyperparamters(0,7,11,11)
    #print(params_hyper)
    # Generate the best model
    params_hyper= {'activation': 'tanh', 'hidden_layer_sizes': (10, 9), 'learning_rate_init': 0.001, 'max_iter': 1000, 'solver': 'adam'} 
    mod= rn.Generate_Neuronal_Model_KNFold(params_hyper,20,PATH_MODELS)
    #mod= rn.Generate_Neuronal_Model_KNFold(activation,neu1,neu2,N_SP,PATH_MODELS,100000)

if __name__ == "__main__":
    #dataset_managment(4)
    df=DataEngineering(config['PATH_DATASET'],config['PATH_OUTPUT_TRAINING'],config['DROP_COLUMNS'],config['TARGET_ML'])
    Training_Process(df,config['TARGET_ML'],config['PATH_MODELS'])

    '''
    {'solver': 'adam', 'max_iter': 10000, 'learning_rate_init': 0.01, 'hidden_layer_sizes': (9, 9), 'activation': 'logistic'}
    Grid Search
    {'activation': 'tanh', 'hidden_layer_sizes': (10, 9), 'learning_rate_init': 0.001, 'max_iter': 1000, 'solver': 'adam'} 
    '''
    '''
    
    import pandas as pd
    import matplotlib.pyplot as plt

    # Leer el DataFrame desde un archivo CSV
    df = pd.read_csv('/Users/jl/Documents/Projects/SV-NASA/database/CASSIOPE/Data_Cassiope/batt_curr.csv')

    # Asegurarse de que la columna de tiempo es de tipo datetime

    # Intenta convertir usando el formato con milisegundos
    df['batt_curr_time'] = pd.to_datetime(df['batt_curr_time'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
    mask = df['batt_curr_time'].isna()
    df.loc[mask, 'batt_curr_time'] = pd.to_datetime(df.loc[mask, 'batt_curr_time'], format="%Y-%m-%d %H:%M:%S", errors='coerce')
    # Ordenar el DataFrame por la columna de tiempo
    df = df.sort_values('batt_curr_time')

    # Graficar la variable en función del tiempo
    plt.figure(figsize=(10, 6))
    plt.plot(df['batt_curr_time'], df['batt_curr'], label='KP')
    plt.xlabel('batt_curr_time')
    plt.ylabel('bat_curr')
    plt.title('bat_curr vs Tiempo')
    plt.legend()
    plt.show()
    '''