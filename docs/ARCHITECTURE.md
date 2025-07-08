# Dynamic Insurance Premium Calculator - Architecture

## System Overview
![Architecture Diagram](https://i.imgur.com/JKQ4W5x.png) *(Example diagram - replace with your own)*

### Core Components

#### 1. Data Collection Module
- **Input Types**:
  - Demographic data (age, gender, location)
  - Risk factors (BMI, smoking status)
  - Coverage requirements
- **Validation**:
  - Range checks (e.g., age 18-100)
  - Categorical value validation

#### 2. Risk Assessment Engine
```python
class RiskModel:
    def assess_risk(self, applicant_data):
        # 1. Preprocess input
        # 2. Apply ML model
        # 3. Generate SHAP explanations
