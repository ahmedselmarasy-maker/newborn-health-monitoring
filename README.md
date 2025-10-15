# Newborn Health Monitoring System

A machine learning-based web application for monitoring newborn health and assessing risk levels using various health parameters.

## Features

- **Health Parameter Monitoring**: Track vital signs and health metrics for newborns
- **Machine Learning Risk Assessment**: Predict health risk levels using trained models
- **Real-time Analysis**: Get instant risk assessments based on input data
- **User-friendly Interface**: Clean and intuitive web interface
- **Data Visualization**: Clear presentation of health status and risk levels

## Health Parameters Monitored

- Gestational age and birth measurements
- Current age, weight, length, and head circumference
- Vital signs (temperature, heart rate, respiratory rate, oxygen saturation)
- Feeding patterns and output measurements
- Jaundice levels and Apgar scores
- Gender, feeding type, immunizations, and reflexes

## Technology Stack

- **Backend**: Python Flask
- **Machine Learning**: Scikit-learn (Decision Tree Classifier)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/newborn-health-monitoring.git
cd newborn-health-monitoring
```

2. Install required packages:
```bash
pip install flask pandas numpy scikit-learn joblib
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## Usage

1. **Input Health Data**: Enter the newborn's health parameters in the web form
2. **Get Risk Assessment**: Click "Assess Risk" to get the ML model's prediction
3. **View Results**: See the risk level (Healthy/At Risk) with confidence percentage

## Model Information

- **Algorithm**: Decision Tree Classifier
- **Features**: 22 health parameters
- **Training Data**: Newborn health monitoring dataset
- **Output**: Binary classification (Healthy/At Risk)

## API Endpoints

- `GET /`: Main application interface
- `POST /predict`: Get risk assessment prediction
- `GET /health`: Health check endpoint
- `GET /model-info`: Get model information

## File Structure

```
newborn-health-monitoring/
├── app.py                              # Main Flask application
├── model.pkl                           # Trained ML model
├── newborn_health_monitoring_with_risk.csv  # Training dataset
├── notebook.ipynb                      # Data analysis notebook
├── static/
│   ├── css/style.css                  # Styling
│   └── js/script.js                   # Frontend JavaScript
├── templates/
│   └── index.html                      # Main HTML template
└── .gitignore                          # Git ignore file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This application is for educational and research purposes. Always consult with healthcare professionals for actual medical decisions.
