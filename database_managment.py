
from database import DATABASE_integration
from database_DSCOVR import DataBase_DSCOVR
from database_MADRIGAL import DataBase_MADRIGAL
from database_CASSIOPE import DataBase_CASSIOPE
import yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def dataset_managment(case):
    """
    Manage datasets based on the value of 'case'.

    Parameters:
    case (int): Determines the source of the data and the operation to be performed.
        - If case == 1: Generate Data from DSCOVR_PlasMAG.
        - If case == 2: Get Data from Madrigal DataSets.
        - If case == 3: Get Data from Cassiope DataSets.
        - If case == 4: Integrate CASSIOPE and DSCOVR Datasets.
        - If case is none of the above, the function does nothing.
    """
    if case==1:
        #1.0 Generate Data from DSCOVR_PlasMAG'
        db= DataBase_DSCOVR(config['filepath_DSCOVR'])
        db.generate_DataSet(config['params_model_DSCOVR'],'bt',config['output_filename'])
    if case==2:  
        #2.0 et Data from Madrigal DataSets
        dm= DataBase_MADRIGAL(config['user_fullname'],config['user_email'],config['web_madrigal'],config['user_afiliation'])
        dm.load_data_from_madrigera_web(config['instrument_code'],config['params'],config['start_date'],config['end_date'], config['output_filename_mad'])
    if case==3:
        #3.0 Get Data from Cassiope DataSets
        dc= DataBase_CASSIOPE(config['input_filename_cass'],config['output_filename_cass'],config['settings_target'])
        dc.generate_DataSet(int(config['settings_target_value']),config['params_cass'][0],10)
    if case==4:  
        #4.0 Integrations CASSIOPE and DSCOVR Datasets
        di= DATABASE_integration()
        di.dataset_generation_DS_CASS(config['PATH_FILE_CASS'],config['PATH_FILE'],config['PATH_OUTPUT'],config['TIME_DELTA'])
    else:
        pass    