from flask import Flask, jsonify

app = Flask(__name__)

# Hardcoded JSON object
hardcoded_json_object = {
    "ART": 13436,
    "CPM": 3,
    "EPM": 10,
    "ExPM": 3,
    "HTTP": 23370,
    "CPU":89,
    "MEM":90
}

@app.route('/get_hardcoded_object', methods=['GET'])
def get_hardcoded_object():
    return jsonify(hardcoded_json_object)

if __name__ == '__main__':
    app.run(debug=True)
