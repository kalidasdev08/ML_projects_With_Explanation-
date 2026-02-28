# Resume Screening AI 🎯

An AI-powered resume screening tool that matches candidate resumes with job descriptions using Natural Language Processing (NLP) techniques.

## Problem Statement

HR teams manually screen thousands of resumes, which is time-consuming and prone to human error. This AI solution automates the resume screening process by analyzing the match between a candidate's resume and a job description.

## Solution

This project uses **NLP similarity** techniques to:
- Extract keywords from resumes and job descriptions
- Calculate match scores using TF-IDF embeddings
- Perform cosine similarity to determine compatibility
- Identify matching and missing skills
- Provide actionable recommendations

## Features

### Core Features
- **Resume-Job Matching**: Calculate match percentage between resume and job description
- **Keyword Extraction**: Automatically extract important keywords from both documents
- **Skills Analysis**: Identify matching and missing technical skills
- **Recommendations**: Provide actionable suggestions to improve resume matching

### Technical Features
- TF-IDF Vectorization for text embeddings
- Cosine Similarity for match scoring
- N-gram analysis (unigrams and bigrams)
- Technical skills database (50+ skills)
- Clean REST API endpoints

## Project Structure

```
resume_screening/
├── app.py                  # Flask web application
├── resume_screening.py     # Core NLP matching logic
├── train_model.py          # Model training module
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── templates/
│   └── index.html         # Web UI
└── models/
    └── resume_screener.pkl # Saved model (after training)
```

## Installation

1. Clone or download this project
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`. Open this URL in your browser to access the Resume Screening AI interface.

### Using the API

#### Analyze Resume (Full Analysis)
```bash
curl -X POST http://localhost:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Your resume text here...",
    "job_description": "Job description text here..."
  }'
```

#### Quick Match (Percentage Only)
```bash
curl -X POST http://localhost:5000/quick-match \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Your resume text here...",
    "job_description": "Job description text here..."
  }'
```

#### Get Sample Data
```bash
curl http://localhost:5000/sample-data
```

## Example Output

```
🎯 Resume Match Score: 78%

✅ Matching Skills: python, javascript, react, sql, aws
❌ Missing Skills: docker, kubernetes, terraform

📊 Keyword Match: 65%

💡 Recommendations:
• Consider highlighting or acquiring skills: docker, kubernetes, terraform
• Add more keywords from the job description to your resume.
```

## How It Works

### 1. Text Preprocessing
- Convert to lowercase
- Remove special characters
- Normalize whitespace

### 2. TF-IDF Vectorization
- Create TF-IDF vectors for resume and job description
- Use unigrams and bigrams (ngram_range=(1,2))
- Filter stop words

### 3. Cosine Similarity
- Calculate cosine similarity between TF-IDF vectors
- Convert to percentage (0-100%)

### 4. Skills Extraction
- Match against 50+ technical skills database
- Identify matching and missing skills

### 5. Recommendations
- Generate personalized improvement suggestions
- Highlight areas for resume optimization

## Sample Resume & Job Description

### Sample Resume
```
JOHN DOE
Software Engineer

SKILLS:
- Python, Java, JavaScript
- React, Node.js, SQL
- AWS, Docker, Git

EXPERIENCE:
Software Engineer at Tech Corp (2020-Present)
- Developed web applications
- Implemented REST APIs

EDUCATION:
BS Computer Science
```

### Sample Job Description
```
Software Engineer Position

Requirements:
- Python, Java, JavaScript
- React, Node.js
- SQL, MongoDB
- AWS experience

Responsibilities:
- Develop web applications
- Implement APIs
```

## Technology Stack

- **Backend**: Python, Flask
- **ML/NLP**: scikit-learn (TF-IDF, Cosine Similarity)
- **Frontend**: HTML, CSS, JavaScript

## Use Cases

1. **HR Teams**: Quickly screen hundreds of resumes for a position
2. **Job Seekers**: Improve their resumes to match job requirements
3. **Recruiters**: Compare multiple candidates against job descriptions
4. **Career Counselors**: Help clients optimize their resumes

## Extending the Project

### Add More Skills
Edit the `extract_skills()` method in `resume_screening.py` to add more skills to the database.

### Improve Matching
- Use word embeddings (Word2Vec, BERT)
- Add more training data
- Implement deep learning models

### Integrate with ATS
Connect with Applicant Tracking Systems for automated resume parsing.

## License

This project is for educational purposes.

## Author

Created as a machine learning project demonstrating NLP similarity techniques.
