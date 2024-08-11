from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.form:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features from form
        int_features = [float(x) for x in request.form.values()]
        final = [np.array(int_features)]
        print(f"Final: {final}")
        # Make prediction
        prediction = model.predict_proba(final)
        # Log prediction
        percentage = prediction[0][1] * 100  # Assuming you want the probability for the positive class
        print(f"Probability as percentage: {percentage}")
        
        # Return prediction as JSON response
        return jsonify({'prediction': f'{percentage:.2f}%'})
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)  
