
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import glob
import subprocess
import os
import spacepy.pycdf as pycdf
import zipfile
import tempfile
import shutil

class data_integration():
  def __init__(self):
    self.__data=""
  def get_data(self):
    return self.__data
  def get_keys_data(self):
      return self.__data.columns
  def graphic(self,cols_med,title,intervals):
    plt.figure(figsize=(10, 6))
    for column in cols_med:
        plt.plot(self.__data.index, self.__data[column], label=column)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=intervals))
    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
  def load_file_DSCOVR_PlasMAG_EXPDB(self,path_base,column_names,name_file):
    path = path_base+'/*.zip'
    all_data = pd.DataFrame()
    for file in glob.glob(path):
        df = pd.read_csv(file, compression='zip', delimiter=',', parse_dates=[0], na_values='0', header=None)
        all_data = pd.concat([all_data, df], ignore_index=True)
    all_data.columns = column_names
    all_data['bt'] = (all_data['bx_gse']**2 + all_data['by_gse']**2 + all_data['bz_gse']**2)**0.5
    all_data.set_index('time', inplace=True)
    self.__data=all_data.copy()
    path_file = path_base+ str(name_file)
    self.__data.to_csv(path_file, index=True)
  def load_folder_CDS(self, folder_path,label,spv):
        all_dataframes = [] 
        for filename in os.listdir(folder_path):
            if filename.endswith(".zip"):
                full_path = os.path.join(folder_path, filename)
                print(f"Processing {full_path}...")
                self.__load_file_CDS_UnitFileZip(full_path,spv,all_dataframes,label)
        if(len(all_dataframes)>0):
           for i,db in enumerate(all_dataframes):
              path_file = folder_path+ str(spv[i][1])+".csv"
              write_header = not os.path.exists(path_file)
              db.to_csv(path_file, mode='a', header=write_header, index=False)
      
  def __load_file_CDS_UnitFileZip(self, path_zip,spv,all_data,label):
    filename_within_zip = os.path.basename(path_zip).replace('.zip', '')
    temp_dir = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(path_zip, 'r') as zip_ref:
            zip_ref.extract(filename_within_zip, path=temp_dir)
            extracted_filepath = os.path.join(temp_dir, filename_within_zip)
            cdf_file = pycdf.CDF(extracted_filepath)
            for i,lsvar in enumerate(spv):
                df = pd.DataFrame()
                for var in lsvar:
                  df[var] = cdf_file[var][...]
                df['lab']=label  
                if(len(all_data)<i+1):
                   all_data.append(df.copy())
                else:
                   all_data[i] = pd.concat([all_data[i], df.copy()])
            cdf_file.close()
    finally:
        shutil.rmtree(temp_dir)     

  def load_file_CDS(self,path_base):
    path = path_base+'/*.zip'
    cdf_file = pycdf.CDF(path)
    df = pd.DataFrame()
    spv=['batt_curr']
    for var in spv:
      df[var] = cdf_file[var][...]
    print(df)    
    cdf_file.close()
  def load_file_madrigera_db(self,start_date, end_date, output_path):
    cmd = [
        "globalIsprint.py", "--verbose", "--url=http://cedar.openmadrigal.org",
        "--parms=UT1_UNIX,KP,DST,AP3,AP,F10.7",
        f"--output={output_path}",
        "--user_fullname=Jorge+Lozano",
        "--user_email=jorge.fernando.lozano@gmail.com",
        "--user_affiliation=None",
        f"--startDate={start_date}",
        f"--endDate={end_date}",
        "--inst=8100"
    ]
    subprocess.run(cmd)

  def load_file_NOAA(self,filepath,period):
    nc = Dataset(filepath, 'r')
    data = {}
    for var_name in nc.variables.keys():
      data[var_name] = nc.variables[var_name][:]
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'],unit='ms')
    df = df[df['time'].dt.year >= period]
    df.set_index('time', inplace=True)
    self.__data=df.copy()
    nc.close()

  def Process_All_Files(self, folder_path, path_left_file,folder_path_out):
        # Iterar sobre todos los archivos .csv en el directorio folder_path
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv"):
                path_right_file = os.path.join(folder_path, filename)
                self.__Create_Dataset_(path_right_file, path_left_file,folder_path_out) 

  def __Create_Dataset_(self,path_right_file,path_left_file,folder_path):
      df_right = pd.read_csv(path_right_file,sep=',')
      df_left = pd.read_csv(path_left_file,sep=',')
      cols_df_right=df_right.columns[0]
      cols_df_left=df_left.columns[0]
      print(cols_df_right)
      print(cols_df_left)
      df_right[cols_df_right] = pd.to_datetime(df_right[cols_df_right])
      df_left [cols_df_left] = pd.to_datetime(df_left [cols_df_left])
      #df_left ['time_lower'] = df_left [cols_df_left] - pd.Timedelta(minutes=20)
      #df_left['time_upper'] = df_left[cols_df_left] + pd.Timedelta(minutes=20)
      #df_right.set_index(cols_df_right, inplace=True)
      #df_left.set_index(cols_df_left, inplace=True)
      df_right = df_right.sort_values(by=cols_df_right)
      df_left = df_left.sort_values(by=cols_df_left)
      df_merge = pd.merge_asof(df_right, 
                                df_left, 
                                left_on=cols_df_right, 
                                right_on=cols_df_left,
                                direction='nearest',tolerance=pd.Timedelta('20min'))
  
      df_merge = df_merge.drop_duplicates(subset=df_left.columns)
      path_file = folder_path+ "MLDataSet_"+ str(os.path.splitext(os.path.basename(path_right_file))[0])+".csv"
      df_merge.to_csv(path_file, index=False)



