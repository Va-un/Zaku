from src.data_preprocessing import Preprocessing, label_decoding
import pandas as pd
from src.MLC_Classifier import Training_Classifier
from src.Regressor import Training_Regressor



#Example for classification
df = pd.read_csv("Data/gender_classification_v7.csv")
X, y ,Labeler =Preprocessing(df, 'gender')
You = Training_Classifier(X, y,metric="acc") #selection of metric acc,f1
You.Run()
model = You.final_model()
val = model.predict(X)  #labeled value
print(label_decoding(val,Labeler)) # Decoded Values

df = pd.read_csv("Data/Student_Performance.csv")
X, y ,Labeler =Preprocessing(df, 'Performance_Index')
You = Training_Regressor(X, y,metric="mse") #selection of metric acc,f1
You.Run()
model = You.final_model()
val = model.predict(X)
print(val)


