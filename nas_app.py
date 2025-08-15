import os
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash

# Simple Flask app providing minimal API for NAS frontend
app = Flask(__name__)
app.config["MONGO_URI"] = os.environ.get("MONGO_URI", "mongodb://localhost:27017/nas")
mongo = PyMongo(app)

# Users collection
users = mongo.db.users

@app.post('/register')
def register():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({'error': 'username and password required'}), 400
    if users.find_one({'username': username}):
        return jsonify({'error': 'user exists'}), 400
    users.insert_one({'username': username,
                      'password': generate_password_hash(password)})
    return jsonify({'status': 'registered'})

@app.post('/login')
def login():
    data = request.get_json() or {}
    username = data.get('username')
    password = data.get('password')
    user = users.find_one({'username': username})
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'invalid credentials'}), 401
    return jsonify({'status': 'ok'})

@app.post('/credentials')
def save_credentials():
    data = request.get_json() or {}
    username = data.get('username')
    if not username:
        return jsonify({'error': 'username required'}), 400
    creds = {
        'google': data.get('google'),
        'icloud': data.get('icloud')
    }
    users.update_one({'username': username}, {'$set': {'credentials': creds}})
    return jsonify({'status': 'saved'})

@app.get('/files/<filetype>')
def list_files(filetype):
    # Placeholder implementation: real integration with Google Photos and
    # iCloud should fetch remote files. Here we return sample data only.
    if filetype not in {'photos', 'pdfs', 'movies'}:
        return jsonify({'error': 'unsupported type'}), 400
    sample = {
        'photos': ['sample_photo.jpg'],
        'pdfs': ['sample_document.pdf'],
        'movies': ['sample_movie.mp4']
    }
    return jsonify({'files': sample[filetype]})

if __name__ == '__main__':
    app.run(debug=True)
