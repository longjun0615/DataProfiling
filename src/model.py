#%% model functions
from src.utils import *

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
import os
import joblib

def run_train(data, model_path):
    
    pipeline = [('vectorizer', CountVectorizer(ngram_range=(1,2))),
         ('transformer', TfidfTransformer()),
         ('classifier', CalibratedClassifierCV(LinearSVC()))]

    X = data['text'].apply(lambda x:text_prepare(x))
    y = data['class']
    tg = data['Tagname']
    fitted_pipeline = train_and_test(pipeline,X,X,y,y)
    
    with open(model_path, 'wb') as file:
            joblib.dump(fitted_pipeline, file)
    
def run_predict(doc, doc_elm, model_path):
    # doc : column description
    # doc_elm : column name
    with open(model_path, 'rb') as file:
        fitted_pipeline = joblib.load(file)
    class_map=dict(zip(range(55), fitted_pipeline.named_steps['classifier'].classes_))
    
    doc_t = doc_elm + ' ' + doc
    
    """ custom replace rule """
    rep_path = os.path.dirname(os.getcwd()) + '/data/training.xlsx'
    df_replace = pd.read_excel(rep_path, sheet_name='replace', engine='openpyxl')
    ORG =df_replace['org']
    REP = df_replace['rep']
    
    for ind in df_replace.index: 
        doc_t = doc_t.apply(lambda x: x.replace(ORG[ind], REP[ind]))
    doc_t  = doc_t.apply(lambda x:text_prepare(x))
    
    y_pred = fitted_pipeline.predict_proba(doc_t.values)
    
    pred=np.apply_along_axis(filter_threshold,1,y_pred)
    
    max_y = y_pred.max(axis=1)
    max_t = pd.Series(max_y)
    
    final_pred = pd.Series(pred).map(class_map)

    out_df = pd.concat([final_pred, max_t ],1)
    out_df.rename(columns={0:'pred_tag_id',1: 'confidence'}, inplace=True)
    
    tag_path = os.path.dirname(os.getcwd()) + '/data/alltags_v1.csv'

    try:
        tag_df = pd.read_csv(tag_path)
    except:
        tag_df = pd.read_csv('//hpcsmb1.hpc.ford.com/skim78/data/piitagging/alltags_v1.csv')
    
    final = pd.merge(out_df,tag_df, left_on = 'pred_tag_id', right_on = 'Tag ID', how='left')
    final = final.loc[:,['AI/ML Tag','confidence','pred_tag_id']]
    
    return final


