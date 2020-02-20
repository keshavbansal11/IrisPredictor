from flask import Flask,render_template,request
from sklearn.externals import joblib

app=Flask(__name__)
@app.route('/')

def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def show():
    if request.method=='POST':
        sepal_length=request.form['s_length']
        sepal_width=request.form['s_width']
        petal_length=request.form['p_length']
        petal_width=request.form['p_width']
        data=[[float(sepal_length),float(sepal_width),float(petal_length),float(petal_width)]]
        model=joblib.load('iris_model.pkl')
        a=model.predict(data)
        predict_result=''.join(a)
    
    return render_template('result.html',sepal_length=sepal_length,sepal_width=sepal_width,petal_length=petal_length,petal_width=petal_width,predict_result=predict_result)



if __name__=='__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True) 

