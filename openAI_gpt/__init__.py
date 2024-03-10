import os
from flask import Flask, request
from embedding import embedTrainingData
from chatbot import logic

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = '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'

@app.route('/model/train', methods=['POST'])
def modelTraining():
    embedTrainingData()     
    return "training"

@app.route('/model/query', methods=['POST'])
def modelQuering():
    data = request.json
    return logic(data.get("query"))
    # return logic("when was the first computer made?")
    # return

if __name__ == '__main__':
    app.run(debug=True)
