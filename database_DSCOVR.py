import glob
import pandas as pd
#This method integrate all the data from the database DSCOVR_PlasMAG_EXPDB 
PARAMS=['bt','bx_gse','by_gse','bz_gse','time']
class DataBase_DSCOVR:
    def __init__(self,path_base):
        """
        This class serve to process data for DSCOVR_PlasMAG

        Parameters:

        path_base (str): The base path where the data files are located.

        """
        self.data= pd.DataFrame() # This dataset is empty
        self.__path_base=path_base
    def generate_DataSet(self,model_params,target,name_output):
        """
        This method generates a file from the data in the specified path.

        Parameters:

        model_params (list): The names of the columns in the data.
                'time': model_params[0]
                'bx_gse': model_params[1]
                'by_gse': model_params[2]
                'bz_gse': model_params[3]
        target (str): The name of variable of model
        name_output (str): The name of the output file.

        Returns:

        Save information in self.__data and save in output destination
        """
        path = self.__path_base+'/*.zip'
        all_data = pd.DataFrame()
        for file in glob.glob(path):
            df = pd.read_csv(file, compression='zip', delimiter=',', parse_dates=[0], na_values='0', header=None)
            all_data = pd.concat([all_data, df], ignore_index=True)  
        all_data.columns = model_params[0]
        all_data[target] = (all_data[model_params[0][1]]**2 + all_data[model_params[0][2]]**2 + all_data[model_params[0][3]]**2)**0.5
        all_data.set_index(model_params[0][0], inplace=True)
        self.data=all_data.copy()
        path_file = self.__path_base+ str(name_output)
        self.data.to_csv(path_file, index=True)

