# Import necessary libraries
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
from model import get_recommendation

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    try:
        data = request.get_json()
        # Extract the features from the request
        user_age = float(data.get('age', 0))
        user_weight = float(data.get('weight', 0.0))
        user_breed = data.get('breed', '')

        predicted_category = get_recommendation(user_weight, user_age, user_breed)

        # Return the prediction as JSON
        return jsonify({"prediction": predicted_category})

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(debug=True)