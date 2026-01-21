import pandas as pd
import joblib
import os

def load_model():
    """
    Automatically finds and loads the trained model in the Colab environment.
    """
    # Use the filename you used during the training save step
    model_filename = 'cancer_prediction_model.pkl' 
    
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    else:
        # If not found, search for any .pkl file in the current directory
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if pkl_files:
            return joblib.load(pkl_files[0])
        else:
            raise FileNotFoundError("No trained model (.pkl) found in the current directory.")

# Main Execution
if __name__ == "__main__":
    
    # Load the model directly
    try:
        model = load_model()
    except Exception as e:
        print(f"Error: {e}")
        # Stop execution if model is not found
        exit()

    # Define columns exactly as used during training
    columns = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

    print("\n--- Virtual Clinic Test (7 Diverse Cases) ---")

    # Define the 7 test cases
    cases = [
        {'Case': 'Healthy Athletic Young Male', 'Data': [25, 0, 22, 0, 0, 9, 0, 0]},
        {'Case': 'Smoker with Alcohol Consumption', 'Data': [55, 1, 29, 1, 0, 2, 5, 0]},
        {'Case': 'Healthy Lifestyle but High Genetic Risk', 'Data': [30, 0, 24, 0, 2, 5, 1, 1]},
        {'Case': 'High Risk Patient (All Factors)', 'Data': [68, 1, 35, 1, 2, 0, 5, 1]},
        {'Case': 'Severe Obesity Only (No Smoking)', 'Data': [45, 1, 40, 0, 0, 1, 0, 0]},
        {'Case': 'Elderly (80y) but Health-Conscious', 'Data': [80, 0, 23, 0, 0, 6, 0, 0]},
        {'Case': 'Borderline Case (Moderate Risks)', 'Data': [50, 1, 27, 0, 1, 3, 2, 0]}
    ]

    # Table Header
    print(f"{'Case Scenario':<45} | {'Diagnosis':<15} | {'Risk Probability'}")
    print("-" * 85)

    for case in cases:
        # Convert data to DataFrame
        df_test = pd.DataFrame([case['Data']], columns=columns)

        # Predict and get probability
        prediction = model.predict(df_test)[0]
        probability = model.predict_proba(df_test)[0][1]

        # Formatting results
        result_text = "🔴 High Risk" if prediction == 1 else "🟢 Healthy"
        prob_text = f"{probability*100:.1f}%"

        print(f"{case['Case']:<45} | {result_text:<15} | {prob_text}")

    print("-" * 85)
    print("Note: GeneticRisk (0=Low, 1=Medium, 2=High).")
