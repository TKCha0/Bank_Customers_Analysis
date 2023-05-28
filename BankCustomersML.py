import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("BankCustomerData.csv")

df.info()
df.isnull().sum()
df.duplicated().sum()

for column in df.columns:
    if df[column].dtype == "object":
        print(column)
        print(df[column].unique())

df_filtered = df.replace("unknown",np.nan)
df_filtered.drop("poutcome",axis=1,inplace=True)
df_filtered.dropna(inplace=True)

def labelencoder(feature):
    trans = LabelEncoder().fit_transform(df_filtered[feature])
    df_filtered[feature] = trans
    return  

for column in df_filtered.columns:
    if df_filtered[column].dtype == "object":
        labelencoder(column)

X = df_filtered.drop("term_deposit",axis=1)
y = df_filtered["term_deposit"]

X_train, X_test, y_train, y_test = tts(X,y,train_size=0.7,random_state=57)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
dtree.score(X_test,y_test)
dtreey_pred = dtree.predict(X_test)
accuracy_score(y_test, dtreey_pred)

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)
rfcy_pred = rfc.predict(X_test)
accuracy_score(y_test,rfcy_pred)

xgbc = XGBClassifier()
xgbc.fit(X_train,y_train)
xgbcy_pred = xgbc.predict(X_test)
accuracy_score(y_test,xgbcy_pred)

knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knny_pred = knn.predict(X_test)
knn.score(X_test,y_test)
accuracy_score(y_test,knny_pred)





