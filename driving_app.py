# we are going to use flask, a micro web framework
from flask import Flask, jsonify, request
import os
import pickle

# /predict?weather=Clear&road=Dry&light=Daylight&junction=Mid-Block%20(not%20related%20to%20intersection)

# make a flask app
app = Flask(__name__) 

# we need to add two routes (functions that handle requests)
# one for the homepage
@app.route("/", methods=["GET"])
def index():
    # return content and status code
    return "<h1>Welcome to my app</h1>", 200

@app.route("/predict", methods=["GET"])
def predict():
    infile = open('driving_bayes.p', 'rb')
    myb = pickle.load(infile)
    infile.close()

    weather = request.args.get('weather', '')
    road = request.args.get('roadcond', '')
    light = request.args.get('light', '')
    junction = request.args.get('junction', '')
    severity = request.args.get('severity', '')

    prediction = myb.predict([[weather, road, light, junction, severity]])

    if prediction is not None:
        result = {'prediction': prediction}
        return jsonify(result), 200
    else:
        return 'Error making prediction', 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
    # by default, Flask runs on port 5000