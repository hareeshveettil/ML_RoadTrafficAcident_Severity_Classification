import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value


def get_prediction(data,model):
    """
    Predict the class of a given data point.
    """
    
    severity = ["Fatal injury", "Serious Injury", "Slight Injury"]
    
    pred =  model.predict(data)
    return severity[pred[0]-1]