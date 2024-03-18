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

        # Create a DataFrame with the user's input
        user_input = pd.DataFrame({'Weight': [user_weight], 'Age': [user_age]})
        # Add a column for the specified breed
        user_input['Breed_' + user_breed] = 1

        # Ensure column order and structure match the training data
        user_df = user_input.reindex(columns=feature_columns, fill_value=0)

        # Use the trained model to predict preferred foods
        user_pred = model.predict(user_df)

        # Display or use the prediction
        food_category_names = df.columns[df.columns.str.startswith('Preferred Foods_')].tolist()
        predicted_category_index = user_pred.argmax(axis=1)
        predicted_category = food_category_names[predicted_category_index[0]]
        predicted_category = predicted_category.replace('Preferred Foods_', '')

        # Return the prediction as JSON
        return jsonify({"prediction": predicted_category})

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    app.run(debug=True)