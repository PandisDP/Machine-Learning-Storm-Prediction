import subprocess
import madrigalWeb.madrigalWeb
class DataBase_MADRIGAL():
    def __init__(self,user_fullname,user_email,http_path,user_afill):
        self.__user_fullname= user_fullname
        self.__user_email=user_email
        self.__http_path=http_path
        self.__user_afiliation= user_afill
    def get_instruments(self):
        test =  madrigalWeb.madrigalWeb.MadrigalData(self.__http_path)
        instList = test.getAllInstruments()
        return instList
    def load_data_from_madrigera_web(self,instrument_code,params,start_date,end_date,output_path):
        print(self.__http_path)
        print(self.__user_fullname)
        print(self.__user_email)
        print(instrument_code)
        print(params)
        cmd = [
            "globalIsprint.py", 
            "--verbose", 
            f"--url={self.__http_path}",
            f"--parms={params}",
            f"--output={output_path}",
            f"--user_fullname={self.__user_fullname}",
            f"--user_email={self.__user_email}",
            f"--user_affiliation={self.__user_afiliation}",
            f"--startDate={start_date}",
            f"--endDate={end_date}",
            f"--inst={instrument_code}"
        ]
        print(cmd)
        subprocess.run(cmd)
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
        print(cmd)
        subprocess.run(cmd)
