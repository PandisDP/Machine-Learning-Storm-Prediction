from database import data_integration
from NeuronalNetwork import RedNeuronal
from dataset import data_set

def dataset_managment():
  #Demo to laod data from repository
  params = [['batt_curr_time','batt_curr'],['batt_volt_time','batt_volt'],['certo_curr_time','certo_curr'],['fai_curr_time','fai_curr'],
        ['gap_curr_time','gap_curr'],['irm_curr_time','irm_curr'],['mag_a_curr_time','mag_a_curr'],['mag_b_curr_time','mag_b_curr'],
        ['mgf_curr_time','mgf_curr'],['mgf_boom_temp_time','mgf_boom_temp'],['mgf_box_temp_time','mgf_box_temp'],['nms_curr_time','nms_curr'],
        ['rri_curr_time','rri_curr'],['rri_temp_time','rri_temp'],['rx_a_curr_time','rx_a_curr'],['rx_b_curr_time','rx_b_curr'],
        ['sei_curr_time','sei_curr'],['slr_1_volt_time','slr_1_volt'],['slr_2_volt_time','slr_2_volt'],['slr_3_volt_time','slr_3_volt'],
        ['slr_4_volt_time','slr_4_volt'],['slr_curr_time','slr_curr'],['ss_a_curr_time','ss_a_curr'],['ss_b_curr_time','ss_b_curr'],
        ['tr_cmdtrq_x_time','tr_cmdtrq_x'],['tr_cmdtrq_y_time','tr_cmdtrq_y'],['tr_cmdtrq_z_time','tr_cmdtrq_z'],['tx_a_curr_time','tx_a_curr'],['tx_b_curr_time','tx_b_curr']]
  params_model = ["time", "bx_gse", "by_gse", "bz_gse",
                    "fs_4", "fs_5", "fs_6", "fs_7", "fs_8", "fs_9","fs_10", "fs_11", "fs_12", "fs_13", "fs_14", "fs_15",
                    "fs_16", "fs_17", "fs_18", "fs_19", "fs_20", "fs_21","fs_22", "fs_23", "fs_24", "fs_25", "fs_26", "fs_27",
                    "fs_28", "fs_29", "fs_30", "fs_31", "fs_32", "fs_33","fs_34", "fs_35", "fs_36", "fs_37", "fs_38", "fs_39",
                    "fs_40", "fs_41", "fs_42", "fs_43", "fs_44", "fs_45","fs_46", "fs_47", "fs_48", "fs_49", "fs_50", "fs_51",
                    "fs_52", "fs_53"]
  filepath = 'database/'
  db= data_integration()
  #db.load_file_DSCOVR_PlasMAG_EXPDB(filepath,params_model,'db_Plasma_2022-2023.csv')
  # print(db.get_keys_data())
  #cols_med=['bt']
  #db.graphic(cols_med,"Data Repo",300)
  #db.load_file_madrigera_db('01/01/2023', '09/30/2023', 'database_ms/2023a.csv')
  #db.load_folder_CDS('database_ns/',0,params)
  PATH_FOLDER= 'database_ns/'
  PATH_OUTPUT = 'datasetsml/'
  PATH_FILE ='database/db_Plasma_2022-2023.csv'
  db.Process_All_Files(PATH_FOLDER,PATH_FILE,PATH_OUTPUT)
def MachineLearning_Process():
    print("Run Machine Learning")
    PATH = 'datasetsml/MLDataSet_batt_curr.csv'
    PATH_OUTPUT='datasetsml'
    db=data_set()
    df=db.load_database(PATH)
    # Clean NAN
    df=db.drop_high_null_columns(df,0.05)
    # Data Analysis of Correlations
    db.correlations(df,[],PATH_OUTPUT,"Initial")
     # Clear Ouliers
    target_obj= 'lab'# This variable repesent Kp*10 Index
    target_list= df.columns.to_list()
    target_list.pop(0) 
    target_list.pop(2)
    target_list.pop(1)
    print("Run Machine Learning")
    df_balance=db.outliers_clean_base(df,target_list)
    # Balance of data con Sub Sumple Method
    df_balance =db.balance_subsample(df_balance,target_obj)
     # Estadisticas y correlaciones
    lst_p = [0,50,60,70,80]
    lst_a = df_balance.columns.to_list()
    lst_drop=[]
    lst_drop.append(lst_a[0])
    lst_drop.append(lst_a[3])
    lst_a.pop(0) 
    lst_a.pop(2)
    lst_a.pop(1)
    print(lst_a)
    print(lst_drop)
    df2 = df_balance.drop(lst_drop, axis=1)
    #db.multi_picture(df2,lst_p,lst_a,target_obj,PATH_OUTPUT)
  # Create Model of Training and Testing with Red Neuronal
    PATH_MODELS='models'
    rn = RedNeuronal(df2,target_obj,42)
    #This method generate the best hyperparameters of Neuronal Network like activation functiona and neurones of each level
    #params=rn.GridSearch_Hyperparamters(12, 12)
    #p1,p2=rn.Indetify_Hyperparamters_Analysis(2,2)
    #print(p1) # logistic
    #print(p2) #  (1, 5)
    # Generate the best model
    activation= 'logistic'
    neu1=10
    neu2= 10
    N_SP=20
    #mod,esc= rn.Generate_Neuronal_Model(activation,neu1,neu2) # Generate model fitting and escalador
    mod= rn.Generate_Neuronal_Model_KNFold(activation,neu1,neu2,N_SP,PATH_MODELS)
    #y_pred= rn.Generate_Prediction(mod,esc,rut_new)

if __name__ == "__main__":
  #dataset_managment()
  MachineLearning_Process()