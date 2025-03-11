import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = {
    'Height':[170,165,190,180,180,180,173,175,177,163,155],
    'Weight':[60,55,80,95,85,77,70,78,87,65,59],
    'Age':[20,25,22,23,21,20,19,21,22,23,22],    
    'Sex':['M','W','M','M','M','M','M','M','M','W','W'],
}

df = pd.DataFrame(data)

x = df[['Height','Weight']]
y = df['Sex']

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
#accuracy = (y_test,y_pred)


while True:
    N_Height = int(input("Enter Height : "))
    N_Weight = int(input("Enter Weight : "))
    new_person = pd.DataFrame([[N_Height, N_Weight]], columns=['Height', 'Weight'])  # دیتافریم با 2 ویژگی
    prediction = model.predict(new_person)
    print("Woman os Man is : ", prediction[0])
