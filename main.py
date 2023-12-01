from flask import Flask, render_template, request,url_for
import pickle
import numpy as np

app = Flask(__name__)

LoanModel = pickle.load(open('LoanModel.pkl','rb'))

@app.route('/')
def mainpage():
    return render_template('home.html')



@app.route('/loanpredict',methods=['POST'])
def pred_for_loan():
    
    data1 = float(request.form['Married'])  
    data2 =  float(request.form['Self_Employed'])
    data3 =  float(request.form['ApplicantIncome'])
    data4 =  float(request.form['CoapplicantIncome']) 
    data5 =  float(request.form['Loan_Amount_Term'])
    data6 =  float(request.form['Credit_History'])
    data7 =  float(request.form['Property_Area'])
    arr = np.array([data1,data2,data3,data4,data5,data6,data7])
    arr = arr.reshape(1, -1)
    pred = LoanModel.predict(arr)
    return render_template('loanafter.html',data=pred)
if __name__ == "__main__":
    app.run(debug=True)
