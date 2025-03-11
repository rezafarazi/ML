import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = {
    'Height':[170,165,190,180,180,180,173,175,177,163,155],
    'Weight':[60,55,80,95,85,77,70,78,87,65,59],
    'Age':[20,25,22,23,21,20,19,21,22,23,22],    
    'Sex':['M','W','M','M','M','M','M','M','M','W','W'],
}

dt = pd.DataFrame(data)
x = dt[['Height','Weight']]
y = dt['Sex']

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
model = DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

while True:
    N_Height = int(input("Enter new height : "))
    N_Weight = int(input("Enter new weight : "))
    if(N_Height == 0):
        break
    n_preson = pd.DataFrame([[N_Height,N_Weight]],columns = ['Height','Weight'])
    prd = model.predict(n_preson)
    print("Sex is ",prd[0])



# # Data visualization
plt.figure(figsize=(12, 6))

# Create subplots
plt.subplot(1, 2, 1)
plt.scatter(dt['Height'], dt['Weight'], alpha=0.5)
plt.xlabel('Height and weight')
plt.ylabel('Sex')
plt.title('Sex from height and weight')

plt.tight_layout()
plt.savefig('sex-and_height.png')




print(dt)