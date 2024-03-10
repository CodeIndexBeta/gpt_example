import os
from flask import Flask, request
from embedding import embedTrainingData
from chatbot import logic

app = Flask(__name__)

@app.route('/model/train', methods=['POST'])
def modelTraining():
    embedTrainingData()     
    return "training"

@app.route('/model/query', methods=['POST'])
def modelQuering():
    data = request.json
    return logic(data.get("query"))

if __name__ == '__main__':
    app.run(debug=True)
