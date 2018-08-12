# import the nessecary pieces from Flask
from flask import Flask,render_template, request,jsonify,Response
import pickle
import pandas as pd

#Create the app object that will route our calls
app = Flask(__name__)

# Home page endpoint 
@app.route('/', methods = ['GET'])
def home():
    # return '<h1> Hello World </h1><p>My name is Cosma</p>'
    return render_template('home.html')

# person page endpoint 
@app.route('/person', methods = ['GET'])
def person():
    # return '<h1> Hello World </h1><p>My name is Cosma</p>'
    return render_template('people.html')

#  startup page endpoint 
@app.route('/startup', methods = ['GET'])
def startUp():
    # return '<h1> Hello World </h1><p>My name is Cosma</p>'
    return render_template('startup.html')

# investor page endpoint 
@app.route('/investor', methods = ['GET'])
def Investor():
    # return '<h1> Hello World </h1><p>My name is Cosma</p>'
    return render_template('investor.html')

# # mpg endpoint
# @app.route('/mpg', methods = ['GET'])
# def mpg():
#     return render_template('mpg.html')

# # inference endpoint
# model = pickle.load(open('models/linreg.p', 'rb'))
# @app.route('/inference', methods = ['POST'])
# def inference():
#     req = request.get_json()
#     print(req)
#     c,h,w = req['cylinders'],req['horsepower'],req['weight']
#     prediction = list(model.predict([[c,h,w]]))
#     return jsonify({'c':c,'h': h,'w':w,'prediction':prediction[0] })
   
# @app.route('/plot',  methods = ['GET'])
# def plot():
#     df = pd.read_csv('data/cars.csv')
#     data = list(zip(df.mpg,df.weight))
#     return jsonify(data)

#When run from command line, start the server
if __name__ == '__main__':
    app.run(host ='0.0.0.0', port = 3333, debug = True)