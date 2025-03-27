#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'secret_key'

# Load and preprocess the dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=column_names)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # In a real application, you should validate the username and password
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('home'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

@app.route('/home')
def home():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    features = [int(request.form['pregnancies']),
                int(request.form['glucose']),
                int(request.form['bloodpressure']),
                int(request.form['skinthickness']),
                int(request.form['insulin']),
                float(request.form['bmi']),
                int(request.form['age'])]
    prediction = model.predict([features])[0]
    if prediction == 1:
        prediction_text = "The model predicts that you have diabetes. Please consult Dr. Smith for further assistance."
    else:
        prediction_text = "The model predicts that you do not have diabetes. Stay healthy!"
    return render_template('result.html', prediction_text=prediction_text)

@app.route('/logout')
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

