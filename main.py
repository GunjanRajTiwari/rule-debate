from flask import Flask, request
from src.config import setup
from src.query import query_rag

app = Flask(__name__)

@app.route('/')
def healthCheck():
    return 'OK'

@app.route('/query', methods=['POST'])
def query():
    game = request.json['game']
    query = request.json['query']
    return query_rag("In the game of "+ game + ", "+ query)

if __name__ == '__main__':
    # setup()
    app.run()