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
        
        // Send data to server
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            // إعادة تعيين الزر
            submitBtn.disabled = false;
            submitBtn.textContent = 'Assess Risk Level';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Display result
            riskLevelDiv.textContent = `Risk Level: ${data.risk_level}`;
            riskLevelDiv.className = data.risk_level === 'At Risk' ? 'risk-high' : 'risk-low';
            
            confidenceDiv.textContent = `Confidence: ${data.confidence.toFixed(1)}%`;
            
            // عرض التوصيات بناءً على مستوى الخطورة
            recommendationsList.innerHTML = '';
            if (data.risk_level === 'At Risk') {
                recommendationsList.innerHTML = `
                    <li>Consult with a pediatrician immediately</li>
                    <li>Monitor vital signs closely</li>
                    <li>Ensure proper hydration and nutrition</li>
                    <li>Schedule follow-up appointments</li>
                `;
            } else {
                recommendationsList.innerHTML = `
                    <li>Continue regular monitoring</li>
                    <li>Maintain current feeding schedule</li>
                    <li>Schedule routine check-ups</li>
                `;
            }
            recommendationsDiv.classList.remove('hidden');
            
            resultDiv.classList.remove('hidden');
            
            // التمرير إلى النتيجة
            resultDiv.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('Error:', error);
            submitBtn.disabled = false;
            submitBtn.textContent = 'Assess Risk Level';
            alert('Error connecting to server. Please try again.');
        });
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