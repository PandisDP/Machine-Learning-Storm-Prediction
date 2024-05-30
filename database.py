
import pandas as pd
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import spacepy.pycdf as pycdf


class DATABASE_integration():
  def __init__(self):
    self.__data=""
  def get_data(self):
    return self.__data
  def get_keys_data(self):
      return self.__data.columns
  def dataset_generation_DS_CASS(self,folder_path, path_left_file,folder_path_out,time_delta):
        """
        This function integrate two databases
        Parameters:
        folder_path (str): CASSIOPE DataBase
        path_left_file (str): DSCOVR DataBase
        folder_path_out(str): Output Files
        time_delta (str) : delta time of integrations like '20min'
        """
        iter_=0
        for filename in os.listdir(folder_path):
            iter_+=1
            if filename.endswith(".csv"):
                path_right_file = os.path.join(folder_path, filename)
                print('Iter: ',iter_,'--',filename)
                self.__create_Dataset_(path_right_file, path_left_file,folder_path_out,time_delta) 

  def __create_Dataset_(self,path_right_file,path_left_file,folder_path,time_delta):
      df_right = pd.read_csv(path_right_file,sep=',')
      df_left = pd.read_csv(path_left_file,sep=',')
      cols_df_right=df_right.columns[0]
      cols_df_left=df_left.columns[0]
      #Right
      # Intenta convertir usando el formato con milisegundos
      df_right[cols_df_right] = pd.to_datetime(df_right[cols_df_right], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
      # Aquellos que no pudieron convertirse (NaT) intenta sin milisegundos
      mask = df_right[cols_df_right].isna()  # Encuentra donde la conversión falló y resultó en NaT
      df_right.loc[mask, cols_df_right] = pd.to_datetime(df_right.loc[mask, cols_df_right], format="%Y-%m-%d %H:%M:%S", errors='coerce')
      df_right = df_right.dropna(subset=[cols_df_right])
      #Left
      df_left [cols_df_left] = pd.to_datetime(df_left[cols_df_left], format="%Y-%m-%d %H:%M:%S",errors='coerce')
      mask = df_left[cols_df_left].isna()  # Encuentra donde la conversión falló y resultó en NaT
      df_left.loc[mask, cols_df_left] = pd.to_datetime(df_left.loc[mask, cols_df_left], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
      df_left = df_left.dropna(subset=[cols_df_left]) 
      df_right = df_right.sort_values(by=cols_df_right).reset_index(drop=True)
      df_left = df_left.sort_values(by=cols_df_left).reset_index(drop=True)
      df_merge = pd.merge_asof(df_left, df_right, left_on=cols_df_left, right_on=cols_df_right,
                                direction='nearest',tolerance=pd.Timedelta(time_delta))
      df_merge = df_merge.drop_duplicates(subset=df_left.columns)
      path_file = folder_path+ "MLDataSet_"+ str(os.path.splitext(os.path.basename(path_right_file))[0])+".csv"
      print(df_merge.head())
      df_merge.to_csv(path_file, index=False)

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

  


