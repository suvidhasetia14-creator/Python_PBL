import pickle
import pandas as pd
with open(r"C:\Users\DELL\Downloads\logfinal.pkl", "rb") as f:
    model1, scaler, le1, cols = pickle.load(f)
print("Model loaded successfully!")


import numpy as np
new_data = input("ENTER ")
new_data=list(map(int,new_data.split()))
print(new_data)
new_data=np.array(new_data).reshape(1,-1)
new_data_scaled = scaler.transform(new_data)
prediction = model1.predict(new_data_scaled)
predicted_label = le1.inverse_transform(prediction)
print("Predicted AQI Bucket:", predicted_label[0])