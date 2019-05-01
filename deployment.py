
# coding: utf-8

#author: ruonan-jia
#every certain time get most recent data from influxdb, 
#lstm model as example

###########

from influxdb import DataFrameClient
from influxdb import InfluxDBClient
from train_model import *
from keras.models import load_model 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings 


if __name__=='__main__':
    db = 'windturbine'
#     meas = ['150015', '150021']
    meas = ['150015']
    look_back = 60
    client = InfluxDBClient(url_of_server, 8086, username, passwd, db )
    while(True):
        json_body = []
        for wb in meas:
            df = client.query('SELECT * FROM "%s" LIMIT %s'%(wb,look_back))
            res = df.raw
            labels = res['series'][0]['columns']
            values = res['series'][0]['values']
            df = pd.DataFrame.from_records(values,columns = labels)
            df = df.apply(pd.to_numeric, errors='ignore')
            df['wman_state'] = df['wman_state'].map(conv2Noneflag)            
            if(df['wman_state'].isnull().values.any()):
                print('data contains NaN!')
                break
                
            df_sel = df.drop(["time", 'temperature','relative_humidity','pressure',
                     'dew_temperature', 'wman_state','wtur_flt_main'], 1)  

            f = open('tmp/column_min26.txt','r')  
            a = f.read()  
            column_min = eval(a)  
            f.close() 
            f = open('tmp/column_max26.txt','r')  
            a = f.read()  
            column_max = eval(a)  
            f.close() 
            
            for i in range(df_sel.shape[1]):
                column = list(df_sel)[i]
                df_sel.loc[:,column] = (df_sel.loc[:,column] - column_min[column])/(column_max[column] - column_min[column])
            
            x_data = df_sel.iloc[:,:].values
            x_data = x_data[np.newaxis, ...]
            
            model = load_model('tmp/lstm_weights_171225_1059.hdf5')
            y_pred = model.predict(x_data)
            if y_pred[0,0] < y_pred[0,1]:
                json_body.append(
                {
                    "measurement": "diagnose_test",
                    "tags": {
                        "turbine": wb
                    },
                    "time": df.loc[look_back-1,'time'],
                    "fields": {
                        "fault": 133
                    }
                }
                    )
            
        client.write_points(json_body)
        #update results for every 5mins
        #time.sleep(5*60*1000)
        break

