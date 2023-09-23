import os
import pickle

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split

RANDOM_SEED = 1234

data =load_iris()

X = data['data']
Y = data['target']

x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.3,ramdom_state=RANDOM_SEED)

model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
model.fit(x_train,y_train)

print(f"Accuracy : {accuracy_score(y_test,model.predict(x_test))}")
print(classification_report(y_test,model.predict(x_test)))

os.makedirs("./build",exist_ok=True)
pickle.dump(model,open('./build/model.pkl','wb'))
