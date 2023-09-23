from flask import Flask
import json

app = Flask(__name__)

@app.route("/predict",methods=["POST","GET"])
def inference():
    return json.dumps({'hello':'world'}),200 ## http status code)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=55000)
