from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load LabelEncoder
enc = load('label_encoder.joblib')


loaded_linearSVC_model = load('linearSVC_model.joblib')

def predict(model, text):
    text = pd.Series(text)
    pred = model.predict(text)
    print(enc.classes_[pred][0])

# Example predictions
predict(loaded_linearSVC_model, "I'm very happy to see you")
predict(loaded_linearSVC_model, "Do you really think it is possible? Really, are you sure about it?")