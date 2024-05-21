import yaml
from NeuronalNetwork import RedNeuronal
from dataset import data_set
from database_managment import dataset_managment

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def DataEngineering(PATH,PATH_OUTPUT,drop_columns=[],target_obj='lab'):
    print("Data Engineering Process")
    db=data_set()
    df=db.load_database(PATH,drop_columns)
    # Clean NANclear
    print("Drop Null Columns  of 5% of data")
    df=db.drop_high_null_columns(df,0.05)
    # Data Analysis of Correlations
    print("Correlation Analysis of Data")
    db.correlations(df,[],PATH_OUTPUT,"Initial")
    # Clear Ouliers
    target_list= df.columns.to_list()
    print("Process of Outliers")
    df_balance=db.outliers_clean_base(df,target_list[2:-1])
    # Balance of data con Sub Sumple Method
    df_balance =db.balance_subsample(df_balance,target_obj)
    # Estadisticas y correlaciones
    lst_p = [0,50,60,70,80]
    lst_a = df_balance.columns.to_list()
    db.multi_picture(df_balance,lst_p,lst_a,target_obj,PATH_OUTPUT) 
    return df_balance
def Training_Process(df,target_obj,PATH_MODELS):
    print("Training Process")
    # Create Model of Training and Testing with Red Neuronal
    rn = RedNeuronal(df,target_obj,42)
    #This method generate the best hyperparameters of Neuronal Network like activation functiona and neurones of each level
    #p1,p2=rn.Indetify_Hyperparamters_Analysis(2,2)
    #print(p1) # logistic
    #print(p2) #  (1, 5)
    # Generate the best model
    activation= 'logistic'
    neu1=10
    neu2= 10
    N_SP=2
    mod= rn.Generate_Neuronal_Model_KNFold(activation,neu1,neu2,N_SP,PATH_MODELS,100000)

if __name__ == "__main__":
  dataset_managment(5)
  df=DataEngineering(config['PATH_DATASET'],config['PATH_OUTPUT_TRAINING'],config['DROP_COLUMNS'],config['TARGET_ML'])
  Training_Process(df,config['TARGET_ML'],config['PATH_MODELS'])