from flask import Flask,render_template,request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def diabetes():
	if(request.method=='GET'):
		return render_template('form.html')
	else:
		df=pd.read_csv('datasets_228_482_diabetes.csv')
		X=df.drop(['Outcome'],axis=1)
		y=df['Outcome']
		X_train,_test,y_train,y_test=train_test_split(X,y,test_size=0.10, random_state=5)
		#fit the model now
		reg=LogisticRegression()
		reg.fit(X_train,y_train)
		Pregnancies=int(request.form['Pregnancies'])
		Glucose=int(request.form['Glucose'])
		BloodPressure=int(request.form['BloodPressure'])
		SkinThickness=int(request.form['SkinThickness'])
		Insulin=int(request.form['Insulin'])
		BMI=int(request.form['BMI'])
		DiabetesPedigreeFunction=float(request.form['DiabetesPedigreeFunction'])
		Age=int(request.form['Age'])
		new=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,
			BMI,DiabetesPedigreeFunction,Age]])
		y_pred=reg.predict(new)
		return render_template('result.html',y_pred=y_pred)

if __name__=='__main__':
	app.run(debug=True)




