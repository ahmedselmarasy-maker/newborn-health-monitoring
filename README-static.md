# Newborn Health Monitoring System

A machine learning-based web application for monitoring newborn health and assessing risk levels using various health parameters. This version is optimized for static hosting on Netlify.

## 🌐 Live Demo

[**View Live Application**](https://your-app-name.netlify.app) *(Update with your Netlify URL)*

## Features

- **Health Parameter Monitoring**: Track vital signs and health metrics for newborns
- **Machine Learning Risk Assessment**: Predict health risk levels using JavaScript-based ML model
- **Real-time Analysis**: Get instant risk assessments based on input data
- **User-friendly Interface**: Clean and intuitive web interface
- **Static Hosting**: Optimized for Netlify deployment
- **No Server Required**: Runs entirely in the browser

## Health Parameters Monitored

- Gestational age and birth measurements
- Current age, weight, length, and head circumference
- Vital signs (temperature, heart rate, respiratory rate, oxygen saturation)
- Feeding patterns and output measurements
- Jaundice levels and Apgar scores
- Gender, feeding type, immunizations, and reflexes

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Machine Learning**: Custom JavaScript ML model
- **Hosting**: Netlify (Static Site Hosting)
- **No Backend Required**: Client-side processing only

## Quick Start

### Option 1: Deploy to Netlify (Recommended)

1. **Fork this repository** to your GitHub account
2. **Connect to Netlify**:
   - Go to [Netlify](https://netlify.com)
   - Click "New site from Git"
   - Connect your GitHub account
   - Select this repository
   - Deploy settings are pre-configured in `netlify.toml`
3. **Your app is live!** Netlify will provide you with a URL

### Option 2: Local Development

1. Clone the repository:
```bash
git clone https://github.com/ahmedselmarasy-maker/newborn-health-monitoring.git
cd newborn-health-monitoring
```

2. Serve the files locally:
```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx serve .

# Using PHP
php -S localhost:8000
```

3. Open your browser and navigate to `http://localhost:8000`

## Usage

1. **Input Health Data**: Enter the newborn's health parameters in the web form
2. **Get Risk Assessment**: Click "Assess Risk Level" to get the ML model's prediction
3. **View Results**: See the risk level (Healthy/At Risk) with confidence percentage
4. **Follow Recommendations**: Review the suggested actions based on the assessment

## Model Information

- **Algorithm**: Custom JavaScript Decision Tree-based Model
- **Features**: 22 health parameters
- **Training Data**: Based on medical guidelines and newborn health standards
- **Output**: Binary classification (Healthy/At Risk)
- **Confidence**: Dynamic confidence scoring based on risk factors

## File Structure

```
newborn-health-monitoring/
├── index.html                      # Main application page
├── netlify.toml                    # Netlify configuration
├── static/
│   ├── css/
│   │   └── style.css              # Application styling
│   └── js/
│       ├── script.js              # Original Flask version
│       └── script-static.js       # Static version with ML model
├── templates/
│   └── index.html                 # Flask template (for reference)
├── app.py                         # Flask application (for reference)
├── model.pkl                      # Python ML model (for reference)
├── newborn_health_monitoring_with_risk.csv  # Training data
├── notebook.ipynb                 # Data analysis notebook
└── README.md                       # This file
```

## API Information

This static version doesn't use traditional APIs. Instead, it uses:

- **Client-side ML Model**: JavaScript-based risk assessment
- **Form Processing**: Direct browser-based data processing
- **Real-time Results**: Instant predictions without server calls

## Deployment Options

### Netlify (Recommended)
- ✅ Free hosting
- ✅ Automatic deployments from GitHub
- ✅ Custom domains
- ✅ HTTPS by default
- ✅ Global CDN

### Other Static Hosts
- **Vercel**: `vercel --prod`
- **GitHub Pages**: Enable in repository settings
- **Firebase Hosting**: `firebase deploy`
- **AWS S3 + CloudFront**: Upload files to S3 bucket

## Customization

### Modifying the ML Model
Edit `static/js/script-static.js` to adjust:
- Risk thresholds
- Weight calculations
- Additional risk factors
- Confidence scoring

### Styling Changes
Modify `static/css/style.css` for:
- Color schemes
- Layout adjustments
- Responsive design
- Custom animations

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

**Built with ❤️ for newborn health monitoring**
