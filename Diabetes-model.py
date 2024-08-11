import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
data = pd.read_csv('diabetes.csv')

X = data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

Naive_Bayes = GaussianNB()
Naive_Bayes.fit(X_train, y_train)

y_pred = Naive_Bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy * 100:.2f}%")
sample_data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]  # Replace with a sample that fits your data
prediction = Naive_Bayes.predict_proba(sample_data)
print(f"Prediction for sample data: {prediction}")

pickle.dump(Naive_Bayes, open('model.pkl', 'wb'))
model=pickle.load(open('model.pkl','rb'))



