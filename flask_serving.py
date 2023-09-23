import pickle
import numpy as np
from flask import Flask,jsonify,request

model = pickle.load(open('./build/model.pkl','rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def make_predict():
    request_body = request.get_json(force=True)

    x_test = [request_body['sepal_length'], request_body['sepal_width'],
               request_body['petal_lenght'],request_body['petal_width']]
    
    x_test = np.array(x_test)
    x_test = x_test.reshape(1,-1)

    y_test = model.predict(x_test)

    response_body = jsonify(result=y_test.tolist())

    return response_body

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=55000,debug=True)
