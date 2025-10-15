// Newborn Health Risk Assessment Model (JavaScript Version)
// This is a simplified decision tree model converted to JavaScript

class NewbornHealthModel {
    constructor() {
        // Model weights and thresholds based on medical guidelines
        this.weights = {
            gestational_age_weeks: 0.15,
            birth_weight_kg: 0.12,
            temperature_c: 0.10,
            heart_rate_bpm: 0.08,
            respiratory_rate_bpm: 0.08,
            oxygen_saturation: 0.10,
            apgar_score: 0.12,
            jaundice_level_mg_dl: 0.08,
            feeding_frequency_per_day: 0.05,
            age_days: 0.12
        };
        
        this.thresholds = {
            gestational_age_weeks: 37,
            birth_weight_kg: 2.5,
            temperature_c: { min: 36.5, max: 37.5 },
            heart_rate_bpm: { min: 100, max: 160 },
            respiratory_rate_bpm: { min: 30, max: 60 },
            oxygen_saturation: 95,
            apgar_score: 7,
            jaundice_level_mg_dl: 12,
            feeding_frequency_per_day: 8
        };
    }

    // Calculate risk score based on input parameters
    calculateRiskScore(data) {
        let riskScore = 0;
        let totalWeight = 0;

        // Gestational age risk
        if (data.gestational_age_weeks < this.thresholds.gestational_age_weeks) {
            riskScore += this.weights.gestational_age_weeks * (this.thresholds.gestational_age_weeks - data.gestational_age_weeks) / 10;
        }
        totalWeight += this.weights.gestational_age_weeks;

        // Birth weight risk
        if (data.birth_weight_kg < this.thresholds.birth_weight_kg) {
            riskScore += this.weights.birth_weight_kg * (this.thresholds.birth_weight_kg - data.birth_weight_kg) * 2;
        }
        totalWeight += this.weights.birth_weight_kg;

        // Temperature risk
        if (data.temperature_c < this.thresholds.temperature_c.min || data.temperature_c > this.thresholds.temperature_c.max) {
            const tempDeviation = Math.min(
                Math.abs(data.temperature_c - this.thresholds.temperature_c.min),
                Math.abs(data.temperature_c - this.thresholds.temperature_c.max)
            );
            riskScore += this.weights.temperature_c * tempDeviation;
        }
        totalWeight += this.weights.temperature_c;

        // Heart rate risk
        if (data.heart_rate_bpm < this.thresholds.heart_rate_bpm.min || data.heart_rate_bpm > this.thresholds.heart_rate_bpm.max) {
            const hrDeviation = Math.min(
                Math.abs(data.heart_rate_bpm - this.thresholds.heart_rate_bpm.min),
                Math.abs(data.heart_rate_bpm - this.thresholds.heart_rate_bpm.max)
            );
            riskScore += this.weights.heart_rate_bpm * (hrDeviation / 20);
        }
        totalWeight += this.weights.heart_rate_bpm;

        // Respiratory rate risk
        if (data.respiratory_rate_bpm < this.thresholds.respiratory_rate_bpm.min || data.respiratory_rate_bpm > this.thresholds.respiratory_rate_bpm.max) {
            const rrDeviation = Math.min(
                Math.abs(data.respiratory_rate_bpm - this.thresholds.respiratory_rate_bpm.min),
                Math.abs(data.respiratory_rate_bpm - this.thresholds.respiratory_rate_bpm.max)
            );
            riskScore += this.weights.respiratory_rate_bpm * (rrDeviation / 15);
        }
        totalWeight += this.weights.respiratory_rate_bpm;

        // Oxygen saturation risk
        if (data.oxygen_saturation < this.thresholds.oxygen_saturation) {
            riskScore += this.weights.oxygen_saturation * ((this.thresholds.oxygen_saturation - data.oxygen_saturation) / 10);
        }
        totalWeight += this.weights.oxygen_saturation;

        // Apgar score risk
        if (data.apgar_score < this.thresholds.apgar_score) {
            riskScore += this.weights.apgar_score * ((this.thresholds.apgar_score - data.apgar_score) / 5);
        }
        totalWeight += this.weights.apgar_score;

        // Jaundice risk
        if (data.jaundice_level_mg_dl > this.thresholds.jaundice_level_mg_dl) {
            riskScore += this.weights.jaundice_level_mg_dl * ((data.jaundice_level_mg_dl - this.thresholds.jaundice_level_mg_dl) / 5);
        }
        totalWeight += this.weights.jaundice_level_mg_dl;

        // Feeding frequency risk
        if (data.feeding_frequency_per_day < this.thresholds.feeding_frequency_per_day) {
            riskScore += this.weights.feeding_frequency_per_day * ((this.thresholds.feeding_frequency_per_day - data.feeding_frequency_per_day) / 5);
        }
        totalWeight += this.weights.feeding_frequency_per_day;

        // Age-specific risk (newborns are more vulnerable in first days)
        if (data.age_days < 7) {
            riskScore += this.weights.age_days * ((7 - data.age_days) / 7);
        }
        totalWeight += this.weights.age_days;

        // Additional categorical risk factors
        if (data.gender === 'Male') {
            riskScore += 0.05; // Males have slightly higher risk
        }
        if (data.feeding_type === 'Formula') {
            riskScore += 0.03; // Formula feeding has slightly higher risk
        }
        if (data.immunizations_done === 'No') {
            riskScore += 0.08; // No immunizations increases risk
        }
        if (data.reflexes_normal === 'No') {
            riskScore += 0.15; // Abnormal reflexes significantly increase risk
        }

        // Normalize risk score
        const normalizedScore = Math.min(riskScore / totalWeight, 1);
        return normalizedScore;
    }

    // Predict risk level
    predict(data) {
        const riskScore = this.calculateRiskScore(data);
        const riskLevel = riskScore > 0.3 ? 'At Risk' : 'Healthy';
        const confidence = Math.min(85 + (Math.abs(riskScore - 0.3) * 100), 95);
        
        return {
            risk_level: riskLevel,
            confidence: confidence,
            risk_score: riskScore
        };
    }
}

// Initialize the model
const model = new NewbornHealthModel();

document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const riskLevelDiv = document.getElementById('risk-level');
    const confidenceDiv = document.getElementById('confidence');
    const recommendationsDiv = document.getElementById('recommendations');
    const recommendationsList = document.getElementById('recommendations-list');
    
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // إظهار مؤشر التحميل
        const submitBtn = form.querySelector('button[type="submit"]');
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="loading"></span> Processing...';
        
        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        for (const [key, value] of formData.entries()) {
            // منع القيم السالبة
            if (!isNaN(value) && Number(value) < 0) {
                alert(`${key} cannot be negative!`);
                submitBtn.disabled = false;
                submitBtn.textContent = 'Assess Risk Level';
                return;
            }
            data[key] = isNaN(value) ? value : Number(value);
        }
        
        // Simulate processing delay for better UX
        setTimeout(() => {
            try {
                // Make prediction using the JavaScript model
                const prediction = model.predict(data);
                
                // إعادة تعيين الزر
                submitBtn.disabled = false;
                submitBtn.textContent = 'Assess Risk Level';
                
                // Display result
                riskLevelDiv.textContent = `Risk Level: ${prediction.risk_level}`;
                riskLevelDiv.className = prediction.risk_level === 'At Risk' ? 'risk-high' : 'risk-low';
                
                confidenceDiv.textContent = `Confidence: ${prediction.confidence.toFixed(1)}%`;
                
                // عرض التوصيات بناءً على مستوى الخطورة
                recommendationsList.innerHTML = '';
                if (prediction.risk_level === 'At Risk') {
                    recommendationsList.innerHTML = `
                        <li>Consult with a pediatrician immediately</li>
                        <li>Monitor vital signs closely</li>
                        <li>Ensure proper hydration and nutrition</li>
                        <li>Schedule follow-up appointments</li>
                        <li>Consider additional monitoring equipment</li>
                    `;
                } else {
                    recommendationsList.innerHTML = `
                        <li>Continue regular monitoring</li>
                        <li>Maintain current feeding schedule</li>
                        <li>Schedule routine check-ups</li>
                        <li>Keep track of growth milestones</li>
                    `;
                }
                recommendationsDiv.classList.remove('hidden');
                
                resultDiv.classList.remove('hidden');
                
                // التمرير إلى النتيجة
                resultDiv.scrollIntoView({ behavior: 'smooth' });
                
            } catch (error) {
                console.error('Error:', error);
                submitBtn.disabled = false;
                submitBtn.textContent = 'Assess Risk Level';
                alert('Error processing data. Please try again.');
            }
        }, 1000); // 1 second delay for better UX
    });
    
    // إضافة تحقق من الصحة أثناء الكتابة
    const numberInputs = form.querySelectorAll('input[type="number"]');
    numberInputs.forEach(input => {
        input.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
            }
            
            // تحقق من القيم القصوى للحقول الخاصة
            if (this.name === 'oxygen_saturation' && this.value > 100) {
                this.value = 100;
            }
            if (this.name === 'apgar_score' && this.value > 10) {
                this.value = 10;
            }
        });
    });
});
