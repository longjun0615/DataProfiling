# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:02:16 2021

@author: SKIM78
"""
import time
import requests  
import pandas as pd
import urllib

def api_on(url):
    try:
        urllib.request.urlopen(url, timeout=1)
        return True
    except urllib.error.URLError as err: 
        return False
    
url = 'http://mach1.hpc.ford.com/skim78/pii-tagging'
#url = 'http://localhost:8080'

http_proxy  = "http://internet.ford.com:83"
https_proxy = "http://internet.ford.com:83"
headers = {
#    'user-agent': 'networking.istio.io/v1alpha3',
    'accept': 'application/json',
    'Content-Type': 'application/json'
}
proxyDict = { 
              "http"  : http_proxy, 
              "https" : https_proxy
            }

r = requests.get(url+'/', proxies=proxyDict)
#%% predict
post = '/predict/'
num_try = 10

ts = time.time()

for i in range(num_try):
        
    new_data = {
            "in_data": 
                [
                    {
                        "table_name": "string",
                        "database_name": "string",
                        "column_name": "cds_id",
                        "description_p": "ford email name"
                    },
                            
                    {
                        "table_name": "string",
                        "database_name": "string",
                        "column_name": "vin_num",
                        "description_p": "unique vehicle number"
                    }
                ]
            }
    
    res_predict = requests.post(url+post, json=new_data, headers=headers)
#    print(res_predict.content)
    df = pd.DataFrame.from_dict(res_predict.json(), orient='index')
print(time.time()-ts)
#%% predict
post = '/predict/'
num_data = 4000
single_data = {
                "table_name": "string",
                "database_name": "string",
                "column_name": "cds_id",
                "description_p": "ford email name"
                }
data = {
        "in_data": 
            [single_data for i in range(num_data)]
        }

success=False
num_try = 0
while success is False:
    if num_try == 20:
        print('too many tries')
        break
    try:
        res_predict = requests.post(url+post, json=data)#, headers=headers)
        success = True
        print(num_try)
    except:
        num_try += 1
        continue
    
#res_predict = requests.post(url+post, json=new_data, headers=headers)
#res_predict = requests.post(url+post, json=new_data)
#print(res_predict.content[:2])
df = pd.DataFrame.from_dict(res_predict.json(), orient='index')

#%% predict_csv
import pandas as pd

post = '/predict_csv/'

""" csv file path with two columns which are "column_name" and "description" """
fpath = 'C:/Users/skim78/Desktop/repos/PIITagging/data/test.csv'
fpath = 'C:/Users/skim78/Documents/data/piitagging/EDC_results_input_api_1.csv'

new_file = {'csv_file': open(fpath, 'rb')}

ts=time.time()
res_predict_csv = requests.post(url+post, files=new_file)
#print(res_predict_csv.content)

#df = pd.read_json(res_predict_csv.json(), orient='index')
df = pd.DataFrame.from_dict(res_predict_csv.json(), orient='index')
tf=time.time()
print('{} sec'.format(tf-ts))
#%% predict_csv_columns
import pandas as pd

post = '/predict_csv_columns/'

""" csv file with columns_in """
#fpath = 'C:/Users/skim78/Desktop/repos/PIITagging/data/test.csv'
fpath = r'C:\Users\skim78\Documents\data\piitagging/DF_HIVE_DSC10598_CVMS_LZ_DB_2021-03-22-18-27-11_JDANNY_2021-03-29-17-25-11.csv'
column_names = {'columns_in': ['Column Name'],
                'columns_out': ['AI TAG NAME', 'CONFIDENCE']}

fpath = 'C:/Users/skim78/Documents/data/piitagging/EDC_results_limit.csv'
column_names = {'columns_in': ['COLUMN_NAME'],
                'columns_out': ['TAG_NAME', 'CONFIDENCE', 'TAG_ID']}

test_file = {'csv_file': open(fpath, 'rb')}

ts=time.time()
res_predict_csv = requests.post(url+post, data=column_names, files=test_file)
#df = pd.read_json(res_predict_csv.json())
df = pd.DataFrame.from_dict(res_predict_csv.json(), orient='index')
#df.to_csv('test_tag.csv')
tf=time.time()
print('{} sec'.format(tf-ts))
