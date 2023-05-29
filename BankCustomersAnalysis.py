import pandas as pd
import numpy as np

df = pd.read_csv("BankCustomerData.csv")

print(np.mean(df["pdays"]))
print((np.mean(df["duration"]))/60)
print(np.mean(df["pdays"]))
print(np.median(df["duration"]))

dura = list(df["duration"])
for i in range(len(dura)):
    if dura[i] > 600:
        dura[i] = ">10"
    elif dura[i] >300:
        dura[i] = "5-10"
    else:
        dura[i] = "<5"
df["duration"] = dura

age = list(df["age"])
for i in range(len(age)):
    if age[i] >= 65:
        age[i] = ">65"
    elif age[i] > 45:
        age[i] = "46-64"
    elif age[i] >= 30:
        age[i] = "30-45"    
    else:
        age[i] = "18-29"
df["age"] = age
    
pdays = list(df["pdays"])
for i in range(len(pdays)):
    if pdays[i] > 14:
        pdays[i] = ">2weeks"
    elif pdays[i] >7:
        pdays[i] = ">1week"
    else:
        pdays[i] = "<7"
df["pdays"] = pdays

df.to_csv("BankCustomersAnalysis.csv")


filter1 = (df["age"] == ">65")
df[filter1]









    
    
        