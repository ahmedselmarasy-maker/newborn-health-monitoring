# Newborn Health Monitoring System - Streamlit Version

A machine learning-based web application for monitoring newborn health and assessing risk levels using various health parameters. Built with Streamlit for easy deployment and interactive user experience.

## 🌐 Live Demo

[**View Live Application on Streamlit Cloud**](https://your-app-name.streamlit.app) *(Update with your Streamlit URL)*

## Features

- **Interactive Health Parameter Input**: Easy-to-use sidebar interface
- **Real-time Risk Assessment**: Instant ML-based predictions
- **Visual Health Metrics**: Clear display of current health status
- **Professional Recommendations**: Actionable advice based on risk level
- **Responsive Design**: Works on all devices
- **Medical Guidelines**: Based on established newborn health standards

## Health Parameters Monitored

- **Basic Data**: Gestational age, birth measurements, gender
- **Daily Monitoring**: Current age, weight, length, head circumference
- **Vital Signs**: Temperature, heart rate, respiratory rate, oxygen saturation
- **Feeding Patterns**: Type, frequency, output measurements
- **Additional Factors**: Jaundice levels, Apgar scores, immunizations, reflexes

## Technology Stack

- **Frontend**: Streamlit (Python web framework)
- **Machine Learning**: Scikit-learn (Decision Tree Classifier)
- **Data Processing**: Pandas, NumPy
- **Model Persistence**: Joblib
- **Hosting**: Streamlit Cloud

## Quick Start

### Option 1: Deploy to Streamlit Cloud (Recommended)

1. **Fork this repository** to your GitHub account
2. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitHub account
   - Select this repository
   - Set main file to `app_streamlit.py`
3. **Deploy**: Click "Deploy" and wait for the app to be live
4. **Your app is live!** Streamlit will provide you with a URL

### Option 2: Local Development

1. Clone the repository:
```bash
git clone https://github.com/ahmedselmarasy-maker/newborn-health-monitoring.git
cd newborn-health-monitoring
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app_streamlit.py
```

4. Open your browser and navigate to `http://localhost:8501`

## Usage

1. **Input Health Data**: Use the sidebar to enter newborn health parameters
2. **Review Current Status**: Check the health metrics display
3. **Assess Risk**: Click "Assess Risk Level" button
4. **View Results**: See risk level with confidence percentage
5. **Follow Recommendations**: Review suggested actions based on assessment

## Model Information

- **Algorithm**: Decision Tree Classifier
- **Features**: 22 health parameters
- **Training Data**: Newborn health monitoring dataset
- **Output**: Binary classification (Healthy/At Risk)
- **Confidence**: Dynamic confidence scoring

## File Structure

```
newborn-health-monitoring/
├── app_streamlit.py              # Main Streamlit application
├── app.py                       # Original Flask application
├── requirements.txt              # Python dependencies
├── model.pkl                    # Trained ML model
├── newborn_health_monitoring_with_risk.csv  # Training data
├── notebook.ipynb               # Data analysis notebook
├── static/                      # Static files (for Flask version)
├── templates/                   # HTML templates (for Flask version)
├── index.html                   # Static HTML version
├── netlify.toml                 # Netlify configuration
└── README.md                    # This file
```

## Deployment Options

### Streamlit Cloud (Recommended)
- ✅ Free hosting
- ✅ Automatic deployments from GitHub
- ✅ Custom domains
- ✅ HTTPS by default
- ✅ Easy sharing

### Other Platforms
- **Heroku**: Add Procfile and deploy
- **Railway**: Connect GitHub repository
- **Render**: Deploy as web service
- **AWS**: Use Elastic Beanstalk

## Customization

### Modifying the ML Model
Edit the model training section in `app_streamlit.py` to adjust:
- Risk thresholds
- Feature weights
- Additional risk factors
- Confidence scoring

### UI Customization
Modify the Streamlit interface to:
- Add new input fields
- Change layout
- Customize styling
- Add new visualizations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Medical Disclaimer

⚠️ **Important**: This application is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for actual medical decisions regarding newborn health.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for newborn health monitoring using Streamlit**
