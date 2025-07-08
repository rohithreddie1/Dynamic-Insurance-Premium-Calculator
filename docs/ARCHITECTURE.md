# Dynamic Insurance Premium Calculator - Architecture

## System Overview
<div align="center">
  <img src="https://github.com/rohithreddie1/Dynamic-Insurance-Premium-Calculator/blob/main/docs/architecture.png" width="550" style="border: 1px solid #eee; margin: 20px 0;">
  <p><em>Figure 1: System Architecture Flow</em></p>
</div>

## Core Components

### 1. Data Collection Module
```python
class DataCollector:
    def __init__(self):
        self.valid_regions = ['southwest', 'southeast', 'northwest', 'northeast']
    
    def validate(self, data: dict) -> bool:
        """Ensures all inputs meet requirements"""
        assert 18 <= data['age'] <= 100
        assert data['smoker'] in ['yes', 'no']
        assert data['region'] in self.valid_regions
