"""
Multivariable Linear Regression Project
Assignment 6 Part 3

Group Members:
- Arjun Prabhakaran
- Liam McCreery
- Alexander Schildgen
- Jacob Martinez

Dataset: [World Happiness Ranking]
Predicting: [Country Happiness Score]
Features: [Economy, Family, Freedom, Trust]
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# TODO: Update this with your actual filename
DATA_FILE = 'world_happiness_report.csv'

def load_and_explore_data(filename):
    """
    Load your dataset and print basic information
    
    TODO:
    - Load the CSV file
    - Print the shape (rows, columns)
    - Print the first few rows
    - Print summary statistics
    - Check for missing values
    """
    print("=" * 70)
    print("LOADING AND EXPLORING DATA")
    print("=" * 70)
    
    # Your code here
    df = pd.read_csv(filename)
    print(df.head())
    print(df.shape)
    print(df.columns)
    print(df.describe())


def visualize_data(data):
    """
    Create visualizations to understand your data
    
    TODO:
    - Create scatter plots for each feature vs target
    - Save the figure
    - Identify which features look most important
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
    """
    print("\n" + "=" * 70)
    print("VISUALIZING RELATIONSHIPS")
    print("=" * 70)
    
    # Your code here
    # Hint: Use subplots like in Part 2!
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Happiness vs Features', fontsize=16, fontweight='bold')
    
    # Plot 1: Happiness vs Economy
    axes[0, 0].scatter(data['Economy (GDP per Capita)'], data['Happiness Score'], color='blue', alpha=0.6)
    axes[0, 0].set_xlabel('Economy (GDP per Capita)')
    axes[0, 0].set_ylabel('Happiness Score')
    axes[0, 0].set_title('Happiness vs Economy')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Happiness vs Family
    axes[0, 1].scatter(data['Family'], data['Happiness Score'], color='green', alpha=0.6)
    axes[0, 1].set_xlabel('Family')
    axes[0, 1].set_ylabel('Happiness Score')
    axes[0, 1].set_title('Happiness vs Family')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Happiness vs Freedom
    axes[1, 0].scatter(data['Freedom'], data['Happiness Score'], color='red', alpha=0.6)
    axes[1, 0].set_xlabel('Freedom')
    axes[1, 0].set_ylabel('Happiness Score')
    axes[1, 0].set_title('Happiness vs Freedom')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Leave empty for now (or add another feature later)
    axes[1, 1].scatter(data['Trust (Government Corruption)'], data['Happiness Score'], color='red', alpha=0.6)
    axes[1, 1].set_xlabel('Trust (Government Corruption)')
    axes[1, 1].set_ylabel('Happiness Score')
    axes[1, 1].set_title('Happiness vs Trust (Government Corruption)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('happiness.png', dpi=300, bbox_inches='tight')
    print("\n✓ Feature plots saved as 'happiness.png'")
    plt.show()


def prepare_and_split_data(data):
    """
    Prepare X and y, then split into train/test
    
    TODO:
    - Separate features (X) and target (y)
    - Split into train/test (80/20)
    - Print the sizes
    
    Args:
        data: your DataFrame
        feature_columns: list of feature column names
        target_column: name of target column
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 70)
    print("PREPARING AND SPLITTING DATA")
    print("=" * 70)
    
    # Your code here
    
    feature_columns = ['Economy (GDP per Capita)', 'Family', 'Freedom', 'Trust (Government Corruption)']
    X = data[feature_columns]
    y = data['Happiness Score']
    
    print(f"\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print(f"\nFeature columns: {list(X.columns)}")

    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    
    X_train = X.iloc[:250]
    X_test = X.iloc[250:]
    y_train = y.iloc[:250]
    y_test = y.iloc[250:]

    print(f"\n=== Data Split (Matching Unplugged Activity) ===")
    print(f"Training set: {len(X_train)} samples (first 250 rows)")
    print(f"Testing set: {len(X_test)} samples (last 60 rows)")
    print(f"\nNOTE: We're NOT scaling features here so coefficients are easy to interpret!")
    
    return X_train, X_test, y_train, y_test, feature_columns



def train_model(X_train, y_train, feature_columns):
    """
    Train the linear regression model
    
    TODO:
    - Create and train a LinearRegression model
    - Print the equation with all coefficients
    - Print feature importance (rank features by coefficient magnitude)
    
    Args:
        X_train: training features
        y_train: training target
        feature_names: list of feature names
        
    Returns:
        trained model
    """
    print("\n" + "=" * 70)
    print("TRAINING MODEL")
    print("=" * 70)
    
    # Your code here
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print(f"\n=== Model Training Complete ===")
    print(f"Intercept: ${model.intercept_:.2f}")
    print(f"\nCoefficients:")
    for name, coef in zip(feature_columns, model.coef_):
        print(f"  {name}: {coef:.2f}")
    
    print(f"\nEquation:")
    equation = f"Price = "
    for i, (name, coef) in enumerate(zip(feature_columns, model.coef_)):
        if i == 0:
            equation += f"{coef:.2f} × {name}"
        else:
            equation += f" + ({coef:.2f}) × {name}"
    equation += f" + {model.intercept_:.2f}"
    print(equation)
    
    return model


def evaluate_model(model, X_test, y_test, feature_columns):
    """
    Evaluate model performance
    
    TODO:
    - Make predictions on test set
    - Calculate R² score
    - Calculate RMSE
    - Print results clearly
    - Create a comparison table (first 10 examples)
    
    Args:
        model: trained model
        X_test: test features
        y_test: test target
        
    Returns:
        predictions
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODEL")
    print("=" * 70)
    
    # Your code here
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    print(f"\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"  → Model explains {r2*100:.2f}% of price variation")
    
    print(f"\nRoot Mean Squared Error: ${rmse:.2f}")
    print(f"  → On average, predictions are off by ${rmse:.2f}")
    
    # Feature importance (absolute value of coefficients)
    print(f"\n=== Feature Importance ===")
    feature_importance = list(zip(feature_columns, np.abs(model.coef_)))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (name, importance) in enumerate(feature_importance, 1):
        print(f"{i}. {name}: {importance:.2f}")
    
    return predictions


def make_prediction(model, feature_columns):
    """
    Make a prediction for a new example
    
    TODO:
    - Create a sample input (you choose the values!)
    - Make a prediction
    - Print the input values and predicted output
    
    Args:
        model: trained model
        feature_names: list of feature names

    'Economy (GDP per Capita)', 'Family', 'Freedom', 'Trust (Government Corruption)']
    """
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)
    
    # Your code here
    # Example: If predicting house price with [sqft, bedrooms, bathrooms]
    # sample = pd.DataFrame([[2000, 3, 2]], columns=feature_names)
    
    # Create input array in the correct order: [Mileage, Age, Brand]
    happiness_features = pd.DataFrame([feature_columns], 
                                 columns=feature_columns)
    predicted_happiness = model.predict(happiness_features)[0]
    
    
    print(f"\n=== New Prediction ===")
    print(f"Economy Score: {feature_columns[0]:.0f}, Family score:{feature_columns[1]}, freedom score:{feature_columns[2]}, trust score:{feature_columns[3]}")
    print(f"Predicted price: ${predicted_happiness:,.2f}")
    
    return predicted_happiness



if __name__ == "__main__":
    # Step 1: Load and explore
    data = load_and_explore_data(DATA_FILE)
    
    # Step 2: Visualize
    visualize_data(data)
    
    # Step 3: Prepare and split
    X_train, X_test, y_train, y_test = prepare_and_split_data(data)
    
    # Step 4: Train
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # Step 6: Make a prediction, add features as an argument
    make_prediction(model)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Analyze your results")
    print("2. Try improving your model (add/remove features)")
    print("3. Create your presentation")
    print("4. Practice presenting with your group!")

