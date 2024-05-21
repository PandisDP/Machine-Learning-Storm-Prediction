import zipfile
import tempfile
import shutil
import os
import pandas as pd
import spacepy.pycdf as pycdf
class DataBase_CASSIOPE:
    def __init__(self,folder_path,settings_target):
        """
        This class serve to process data for DSCOVR_PlasMAG
        Parameters:
        folder_path (str): The base path where the data files are located.
        settings_target (str): This variable indica a target like  kp
        """
        self.folder_path=folder_path
        self.settings_target=settings_target
    def generate_DataSet(self,lbl_settings,params_):
        """
        This function serve to create datasets by params_
        Parameters:
        lbl_settings (str): This variable indica a targer like K= {0,50,60,70,80,90}
        params_ (lst): This a list of parameters
        """
        all_dataframes = [] 
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".zip"):
                full_path = os.path.join(self.folder_path, filename)
                print(f"Processing {full_path}...")
                self.__load_file_CDS_UnitFileZip(full_path,params_,all_dataframes,lbl_settings)
        if(len(all_dataframes)>0):
            for i,db in enumerate(all_dataframes):
                path_file = self.folder_path+ str(params_[i][1])+".csv"
                write_header = not os.path.exists(path_file)
                db.to_csv(path_file, mode='a', header=write_header, index=False)

    def __load_file_CDS_UnitFileZip(self,path_zip,params_,all_data,lbl_settings):
        filename_within_zip = os.path.basename(path_zip).replace('.zip', '')
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(path_zip, 'r') as zip_ref:
                zip_ref.extract(filename_within_zip, path=temp_dir)
                extracted_filepath = os.path.join(temp_dir, filename_within_zip)
                cdf_file = pycdf.CDF(extracted_filepath)
                for i,lsvar in enumerate(params_):
                    df = pd.DataFrame()
                    for var in lsvar:
                        df[var] = cdf_file[var][...]
                    df[self.settings_target]=lbl_settings
                    if(len(all_data)<i+1):
                        all_data.append(df.copy())
                    else:
                        all_data[i] = pd.concat([all_data[i], df.copy()])
                cdf_file.close()
        finally:
            shutil.rmtree(temp_dir)            