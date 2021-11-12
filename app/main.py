"""
any changes, inform Kalyan Thanjavuru (KTHANJA1@ford.com) Data Factory
add logging to track usage of api
ref: https://github.ford.com/NLP/itext_clustering/blob/5be4cb49dd7b0ce25c632671fefca878a5c76e0e/app/main.py
"""

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, parse_obj_as
from starlette.requests import Request
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY

# Machine Learning Packages
import json
from typing import List, Dict, Optional

from src.model import run_train, run_predict
from src.utils import get_model_version

import pickle, joblib
import numpy as np
import pandas as pd
import os
import time
from pathlib import Path
from io import StringIO, BytesIO

global pipeline

app = FastAPI()

class Row(BaseModel):
    table_name: str = None
    database_name: str = None
    column_name: str
    description_p: str
        
class InData(BaseModel):
    in_data: List[Row]

class TrainParams(BaseModel):
    data_path: str
    sheet_name: str
    model_path: str

class ColumnName(BaseModel):
    col_name: str
    
class ColumnNames(BaseModel):
    col_names: List[ColumnName]

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/list_files/", response_class=ORJSONResponse)
async def list_modelfiles(request: Request):
    model_dir = os.path.dirname(os.getcwd()) + '/snapshot'
    lsdir = list(os.scandir(model_dir))
    def name_age(f):
        name = f.name
        age = time.time() - f.stat().st_mtime
        if age < 60:
            age = int(age)
            units = "second(s)"
        elif age < 60*60:
            age = int(age/60)
            units = "minute(s)"
        elif age < 60*60*24:
            age = int(age/(60*60))
            units = "hour(s)"
        else:
            age = int(age/(60*60*24))
            units = "day(s)"
        return "{}: last modified {} {} ago".format(name,age, units)
    files = [f for f in lsdir if f.is_file()]
    return json.dumps([name_age(f) for f in files])


@app.post("/train/", response_class=ORJSONResponse)
async def train(train_params: TrainParams, request: Request):
    
    data_path = train_params.data_path
    sheet_name = train_params.sheet_name
    model_path = train_params.model_path
    
    """ hard coded for testing """
    data_path = os.path.dirname(os.getcwd()) + '/data/training.xlsx'
    sheet_name = 'training1.0.7'
    model_path = os.path.dirname(os.getcwd()) + '/snapshot/linearsvm_pipeline_v1.1.joblib'
    
    try:
        data = pd.read_excel(data_path, sheet_name=sheet_name, engine='openpyxl')
    except:
        raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Unable to process file"
                )
        
    run_train(data, model_path)
    return "Model trained and saved"


@app.post("/predict/")#, response_class=ORJSONResponse)
async def predict(in_data: InData, request: Request):
    
    data = json.dumps([s.dict() for s in in_data.in_data])
    data = json.loads(data)
    # get_model_path(version=None) -> latest model or specific version
    model_path = os.path.dirname(os.getcwd()) + '/snapshot/linearsvm_pipeline_v1.1.joblib'
    
    df = pd.DataFrame(data)
    doc = df['description_p']
    doc_elm = df['column_name']

    out_pred = run_predict(doc, doc_elm, model_path)
    out_df = pd.concat([doc_elm, out_pred], axis=1)
    out_df['AI/ML version'] = get_model_version(Path(model_path).stem)
    return out_df.to_dict(orient='index')

@app.post("/predict_csv/")
async def predict_csv(csv_file: UploadFile = File(...)):

    model_path = os.path.dirname(os.getcwd()) + '/snapshot/linearsvm_pipeline_v1.1.joblib'
    try:
        df = pd.read_csv(StringIO(str(csv_file.file.read(), 'utf-8')),
                         sep=',', 
                         usecols=["COLUMN_NAME", "DESCRIPTION"], 
#                         skiprows=1,
                         encoding='utf-8')
    except:
        raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Unable to process file, make sure COLUMN_NAME and DESCRIPTION columns exist"
                )
    
    df['COLUMN_NAME'] = df['COLUMN_NAME'].map(lambda x: '' if pd.isnull(x) else x)
    df['DESCRIPTION'] = df['DESCRIPTION'].map(lambda x: '' if pd.isnull(x) else x)
    
    doc = df['DESCRIPTION']
    doc_elm = df['COLUMN_NAME']

    out_pred = run_predict(doc, doc_elm, model_path)
    
    out_df = pd.concat([df, out_pred], axis=1)
    out_df['AI/ML version'] = get_model_version(Path(model_path).stem)
    return out_df.to_dict(orient='index')

@app.post("/predict_csv_columns/")
async def predict_csv_columns(columns_in: List[str] = Form(...), columns_out: List[str] = Form(...), csv_file: UploadFile = File(...)):
    """
    columns_in : name of columns used for prediction \n
    columns_out : name of columns for output. \n
    \t Rows of first column name entry(required) will be tag name results \n
    \t Rows of second column name(optional) will be confidence of prediction \n
    \t Rows of third column name(optional) will be tag id \n
    \t If column name exists in original csv, output uses existing column name. \n
    \t if not, output will be added as new column at the end. \n
    \t model version will be added to column "AI/ML version". \n
    csv_file : csv_file for prediction
    """
    
    if ',' in columns_in[0]:
        columns_in = columns_in[0].split(',')
    
    if columns_out and ',' in columns_out[0]:
        columns_out = columns_out[0].split(',')
        
    model_path = os.path.dirname(os.getcwd()) + '/snapshot/linearsvm_pipeline_v1.1.joblib'
    
    try:
        df = pd.read_csv(StringIO(str(csv_file.file.read(), 'utf-8')),
                         sep=',',
                         encoding='utf-8')
    except:
        raise HTTPException(
                status_code=HTTP_422_UNPROCESSABLE_ENTITY, detail="Unable to process file"
                )
    
    df.fillna('', inplace=True)
    i=0
    doc_elm=''
    for col_name in columns_in:
        df[col_name] = df[col_name].map(lambda x: '' if pd.isnull(x) else x)
        if i == 0:
            doc = df[col_name]
        else:
            doc_elm += df[col_name] + ' '
        i+=1

    out_pred = run_predict(doc, doc_elm, model_path)
    try:
        for i, column_out in enumerate(columns_out):
            df[column_out] = out_pred.iloc[:,i]
        out_df = df.copy()
    except:
        out_df = pd.concat([df, out_pred], axis=1)

    out_df['AI/ML version'] = get_model_version(Path(model_path).stem)
    return out_df.to_dict(orient='index')


#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8080)
