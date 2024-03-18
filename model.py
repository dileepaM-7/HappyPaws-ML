import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read the CSV file into a DataFrame
df = pd.read_csv('Final.csv')

# Assuming 'Weight', 'Age', and the 'Breed_' columns are your feature columns
feature_columns = ['Weight', 'Age'] + df.columns[df.columns.str.startswith('Breed_')].tolist()

# Assuming 'Preferred Foods_' columns are your target columns
target_columns = df.columns[df.columns.str.startswith('Preferred Foods_')].tolist()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df[feature_columns], df[target_columns], test_size=0.2, random_state=15)

# Initialize and train the DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(x_train, y_train)

# Get predictions on the test set
y_pred = clf.predict(x_test)

# Print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Define a function to get recommendations
def get_recommendation(weight, age, breed):
    # Create a DataFrame with the user's input
    user_input = pd.DataFrame({'Weight': [weight], 'Age': [age]})
    # Add a column for the specified breed
    user_input['Breed_' + breed] = 1
    # Ensure column order and structure match the training data
    user_df = user_input.reindex(columns=feature_columns, fill_value=0)
    # Use the trained model to predict preferred foods
    user_pred = clf.predict(user_df)
    # Display or use the prediction
    food_category_names = df.columns[df.columns.str.startswith('Preferred Foods_')].tolist()
    predicted_category_index = user_pred.argmax(axis=1)
    predicted_category = food_category_names[predicted_category_index[0]]
    predicted_category = predicted_category.replace('Preferred Foods_', '')
    
    return predicted_category