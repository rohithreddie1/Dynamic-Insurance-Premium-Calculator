# insurance_calculator.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import shap
import pickle
from datetime import datetime

class InsurancePremiumCalculator:
    def __init__(self):
        """Initialize the premium calculator system"""
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_order = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
        self.numerical_cols = ['age', 'bmi', 'children']
        self.market_conditions = {
            'inflation_rate': 0.05,
            'competitor_adjustment': -0.02,
            'regional_factors': {
                'southwest': 1.0,
                'southeast': 1.15,
                'northwest': 0.95,
                'northeast': 1.1
            }
        }
        self.regulatory_rules = {
            'max_age_discount': 0.2,
            'min_premium': 5000,
            'prohibited_factors': ['gender']
        }
        self.original_values = None

    def load_data(self, url):
        """Load and preprocess insurance data"""
        df = pd.read_csv(url)
        
        # Encode categorical variables
        categorical_cols = ['sex', 'smoker', 'region']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            self.label_encoders[col] = le
            
        # Scale numerical features
        self.scaler = StandardScaler()
        df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])
        
        return df
    
    def train_model(self, df):
        """Train the risk assessment model"""
        X = df[self.feature_order]
        y = df["charges"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        print(f"Model trained - Train R²: {train_score:.3f}, Test R²: {test_score:.3f}")
    
    def save_model(self, path):
        """Save the trained model to file"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'feature_order': self.feature_order,
                'numerical_cols': self.numerical_cols,
                'market_conditions': self.market_conditions,
                'regulatory_rules': self.regulatory_rules
            }, f)
    
    def load_model(self, path):
        """Load a pre-trained model from file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoders = data['label_encoders']
            self.scaler = data['scaler']
            self.feature_order = data['feature_order']
            self.numerical_cols = data['numerical_cols']
            self.market_conditions = data.get('market_conditions', {})
            self.regulatory_rules = data.get('regulatory_rules', {})

    def preprocess_input(self, user_data):
        """Prepare user input for prediction"""
        # Store original values for explanation
        self.original_values = user_data.copy()
        
        # Create DataFrame with consistent feature order
        processed = {}
        for feature in self.feature_order:
            processed[feature] = user_data.get(feature, 0)
        
        # Encode categorical variables
        for col, le in self.label_encoders.items():
            if col in processed:
                processed[col] = le.transform([processed[col]])[0]
        
        # Scale numerical features
        numerical_values = [[processed[col] for col in self.numerical_cols]]
        scaled_values = self.scaler.transform(numerical_values)
        
        # Create properly named DataFrame
        input_df = pd.DataFrame([processed], columns=self.feature_order)
        for i, col in enumerate(self.numerical_cols):
            input_df[col] = scaled_values[0][i]
        
        return input_df
    
    def calculate_base_premium(self, user_data):
        """Calculate the base premium based on risk factors"""
        input_df = self.preprocess_input(user_data)
        return self.model.predict(input_df)[0]
    
    def apply_market_adjustments(self, base_premium, user_data):
        """Apply market condition adjustments"""
        adjustments = 1.0
        
        # Regional adjustment
        region = user_data.get('region')
        if region in self.market_conditions['regional_factors']:
            adjustments *= self.market_conditions['regional_factors'][region]
        
        # Economic adjustments
        adjustments *= (1 + self.market_conditions['inflation_rate'])
        adjustments *= (1 + self.market_conditions['competitor_adjustment'])
        
        return base_premium * adjustments
    
    def apply_regulatory_constraints(self, premium, user_data):
        """Ensure premium complies with regulations"""
        # Apply minimum premium
        premium = max(premium, self.regulatory_rules['min_premium'])
        
        # Age-based discounts (example regulation)
        age = user_data.get('age', 30)
        if age > 60:
            discount = min(
                self.regulatory_rules['max_age_discount'],
                (age - 60) * 0.01
            )
            premium *= (1 - discount)
        
        return premium
    
    def generate_explanation(self, user_data):
        """Generate SHAP explanation for the premium"""
        input_df = self.preprocess_input(user_data)
        explainer = shap.Explainer(self.model)
        shap_values = explainer(input_df)
        
        # Convert to readable feature names
        feature_names = {
            'age': 'Age',
            'sex': 'Gender',
            'bmi': 'BMI',
            'children': 'Children',
            'smoker': 'Smoker',
            'region': 'Region'
        }
        
        # Get SHAP values with original values
        explanation = []
        for i, col in enumerate(self.feature_order):
            # Get original value
            if col in self.numerical_cols:
                value = self.original_values[col]
            else:
                le = self.label_encoders.get(col)
                value = le.inverse_transform([input_df.iloc[0, i]])[0] if le else input_df.iloc[0, i]
            
            impact = shap_values.values[0][i]
            
            explanation.append({
                'feature': feature_names.get(col, col),
                'value': value,
                'impact': impact,
                'direction': 'Increases' if impact > 0 else 'Reduces'
            })
        
        # Sort by absolute impact
        return sorted(explanation, key=lambda x: abs(x['impact']), reverse=True)
    
    def calculate_premium(self, user_data):
        """Complete premium calculation pipeline"""
        # Calculate base premium
        base_premium = self.calculate_base_premium(user_data)
        
        # Apply market adjustments
        market_adjusted = self.apply_market_adjustments(base_premium, user_data)
        
        # Apply regulatory constraints
        final_premium = self.apply_regulatory_constraints(market_adjusted, user_data)
        
        # Generate explanation
        explanation = self.generate_explanation(user_data)
        
        return {
            'base_premium': round(base_premium, 2),
            'market_adjusted': round(market_adjusted, 2),
            'final_premium': round(final_premium, 2),
            'explanation': explanation,
            'calculation_date': datetime.now().isoformat(),
            'risk_factors': self.identify_key_risk_factors(user_data)
        }
    
    def identify_key_risk_factors(self, user_data):
        """Identify the most significant risk factors"""
        explanation = self.generate_explanation(user_data)
        return [
            {
                'factor': item['feature'],
                'impact': round(item['impact'], 2),
                'recommendation': self.generate_recommendation(
                    item['feature'],
                    item['value'],
                    item['impact']
                )
            }
            for item in explanation[:3]  # Top 3 factors
        ]
    
    def generate_recommendation(self, factor, value, impact):
        """Generate improvement recommendations based on risk factors"""
        recommendations = {
            'BMI': {
                'condition': lambda v: v > 25,
                'message': "Consider lifestyle changes to reduce BMI (currently {})".format(value)
            },
            'Smoker': {
                'yes': "Quitting smoking could reduce premium by ~30%",
                'no': "Good job maintaining non-smoker status"
            },
            'Age': "Age-based pricing is standard in insurance",
            'Children': {
                'condition': lambda v: v > 0,
                'message': "More children increases family coverage costs"
            }
        }
        
        if factor in recommendations:
            if isinstance(recommendations[factor], dict):
                if 'condition' in recommendations[factor]:
                    if recommendations[factor]['condition'](value):
                        return recommendations[factor]['message']
                elif str(value) in recommendations[factor]:
                    return recommendations[factor][str(value)]
            else:
                return recommendations[factor]
        return "No specific recommendations available"

    def get_user_input(self):
        """Dynamically collect user input with validation"""
        print("\nPlease enter your insurance details:")
        
        user_data = {}
        
        # Age
        while True:
            try:
                age = int(input("Age (18-100): "))
                if 18 <= age <= 100:
                    user_data['age'] = age
                    break
                print("Please enter a valid age between 18-100")
            except ValueError:
                print("Please enter a number")

        # Gender
        while True:
            sex = input("Gender (male/female): ").lower()
            if sex in ['male', 'female']:
                user_data['sex'] = sex
                break
            print("Please enter either 'male' or 'female'")

        # BMI
        while True:
            try:
                bmi = float(input("BMI (10-50): "))
                if 10 <= bmi <= 50:
                    user_data['bmi'] = bmi
                    break
                print("Please enter a valid BMI between 10-50")
            except ValueError:
                print("Please enter a number")

        # Children
        while True:
            try:
                children = int(input("Number of children (0-10): "))
                if 0 <= children <= 10:
                    user_data['children'] = children
                    break
                print("Please enter between 0-10")
            except ValueError:
                print("Please enter a whole number")

        # Smoker
        while True:
            smoker = input("Smoker? (yes/no): ").lower()
            if smoker in ['yes', 'no']:
                user_data['smoker'] = smoker
                break
            print("Please enter 'yes' or 'no'")

        # Region
        regions = ['southwest', 'southeast', 'northwest', 'northeast']
        while True:
            region = input(f"Region ({', '.join(regions)}): ").lower()
            if region in regions:
                user_data['region'] = region
                break
            print(f"Please enter one of: {', '.join(regions)}")

        return user_data

    def display_results(self, result):
        """Display results in a user-friendly format"""
        print("\n" + "="*50)
        print(" INSURANCE PREMIUM CALCULATION RESULTS ".center(50, "="))
        print("="*50)
        
        print(f"\n{'Base Premium:':<25} ₹{result['base_premium']:>10.2f}")
        print(f"{'After Market Adjustments:':<25} ₹{result['market_adjusted']:>10.2f}")
        print(f"{'Final Premium:':<25} ₹{result['final_premium']:>10.2f}")
        
        print("\nTOP RISK FACTORS:")
        for factor in result['risk_factors']:
            print(f"- {factor['factor']:>15}: {factor['recommendation']} (Impact: ₹{abs(factor['impact']):.2f})")
        
        print("\nDETAILED BREAKDOWN:")
        for item in result['explanation']:
            arrow = "↑" if item['direction'] == 'Increases' else "↓"
            print(f"{arrow} {item['feature']+':':<15} {str(item['value']):<15} {item['direction']} premium by ₹{abs(item['impact']):.2f}")

if __name__ == "__main__":
    # Initialize calculator
    calculator = InsurancePremiumCalculator()
    
    # Load and train model (or load pre-trained)
    try:
        calculator.load_model('insurance_model.pkl')
        print("Loaded pre-trained model")
    except FileNotFoundError:
        print("Training new model...")
        df = calculator.load_data("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
        calculator.train_model(df)
        calculator.save_model('insurance_model.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Training new model...")
        df = calculator.load_data("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
        calculator.train_model(df)
        calculator.save_model('insurance_model.pkl')
    
    while True:
        # Get dynamic user input
        user_data = calculator.get_user_input()
        
        # Calculate premium
        try:
            result = calculator.calculate_premium(user_data)
            calculator.display_results(result)
        except Exception as e:
            print(f"\nError calculating premium: {e}")
        
        # Ask to continue
        another = input("\nCalculate another premium? (yes/no): ").lower()
        if another != 'yes':
            print("\nThank you for using our insurance calculator!")
            break
