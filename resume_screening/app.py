"""
Resume Screening AI - Flask Web Application
A web-based resume screening tool that matches resumes with job descriptions.
Supports both text input and file upload (PDF, DOCX, TXT).
Can filter candidates by minimum match threshold.
"""

from flask import Flask, render_template, request, jsonify
import os
import sys

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from resume_screening import ResumeScreener
from file_reader import read_uploaded_file

app = Flask(__name__)

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'docx', 'txt'}

# Initialize the model
model = None


def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def get_model():
    """
    Get or initialize the resume screening model.
    """
    global model
    if model is None:
        model = ResumeScreener()
    
    return model


@app.route('/')
def index():
    """
    Render the main page.
    """
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze resume against job description.
    Supports both text input and file upload.
    """
    try:
        # Check if resume is uploaded as file
        resume_file = request.files.get('resume_file')
        resume_text = ''
        
        if resume_file and resume_file.filename:
            # File upload mode
            if not allowed_file(resume_file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Invalid file format. Please upload PDF, DOCX, or TXT files.'
                }), 400
            
            try:
                resume_text = read_uploaded_file(resume_file)
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error reading file: {str(e)}'
                }), 400
        else:
            # Text input mode
            resume_text = request.form.get('resume', '').strip()
        
        # Get job description (from form or JSON)
        job_description = request.form.get('job_description', '').strip()
        if not job_description:
            # Try to get from JSON
            try:
                json_data = request.get_json()
                if json_data:
                    job_description = json_data.get('job_description', '').strip()
            except:
                pass
        
        # Validate input
        if not resume_text:
            return jsonify({
                'success': False,
                'error': 'Please provide your resume (text or file upload).'
            }), 400
        
        if not job_description:
            return jsonify({
                'success': False,
                'error': 'Please provide the job description.'
            }), 400
        
        # Create a new model and fit with the user's input
        screener = ResumeScreener()
        screener.fit([resume_text], [job_description])
        
        # Get detailed analysis
        analysis = screener.get_detailed_analysis(resume_text, job_description)
        
        # Determine selection status based on threshold
        threshold = float(request.form.get('threshold', 75))
        is_selected = analysis['match_score'] >= threshold
        selection_status = "✅ SELECTED" if is_selected else "❌ NOT SELECTED"
        
        # Prepare response
        response = {
            'success': True,
            'match_score': analysis['match_score'],
            'keyword_match_percentage': analysis['keyword_match_percentage'],
            'skill_match_percentage': analysis['skill_match_percentage'],
            'matching_skills': analysis['matching_skills'],
            'missing_skills': analysis['missing_skills'],
            'resume_keywords': analysis['resume_keywords'],
            'job_keywords': analysis['job_keywords'],
            'recommendations': analysis['recommendations'],
            'threshold': threshold,
            'is_selected': is_selected,
            'selection_status': selection_status
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/analyze-json', methods=['POST'])
def analyze_json():
    """
    Analyze resume against job description (JSON API).
    """
    try:
        data = request.get_json()
        
        resume_text = data.get('resume', '').strip()
        job_description = data.get('job_description', '').strip()
        threshold = data.get('threshold', 75)
        
        # Validate input
        if not resume_text:
            return jsonify({
                'success': False,
                'error': 'Please provide your resume text.'
            }), 400
        
        if not job_description:
            return jsonify({
                'success': False,
                'error': 'Please provide the job description.'
            }), 400
        
        # Create a new model and fit with the user's input
        screener = ResumeScreener()
        screener.fit([resume_text], [job_description])
        
        # Get detailed analysis
        analysis = screener.get_detailed_analysis(resume_text, job_description)
        
        # Determine selection status
        is_selected = analysis['match_score'] >= threshold
        selection_status = "SELECTED" if is_selected else "NOT SELECTED"
        
        # Prepare response
        response = {
            'success': True,
            'match_score': analysis['match_score'],
            'keyword_match_percentage': analysis['keyword_match_percentage'],
            'skill_match_percentage': analysis['skill_match_percentage'],
            'matching_skills': analysis['matching_skills'],
            'missing_skills': analysis['missing_skills'],
            'resume_keywords': analysis['resume_keywords'],
            'job_keywords': analysis['job_keywords'],
            'recommendations': analysis['recommendations'],
            'threshold': threshold,
            'is_selected': is_selected,
            'selection_status': selection_status
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple resumes against a job description and filter by threshold.
    Useful for screening multiple candidates at once.
    """
    try:
        data = request.get_json()
        
        resumes = data.get('resumes', [])  # List of {name, text} or just text
        job_description = data.get('job_description', '').strip()
        threshold = data.get('threshold', 75)
        
        if not job_description:
            return jsonify({
                'success': False,
                'error': 'Please provide the job description.'
            }), 400
        
        if not resumes:
            return jsonify({
                'success': False,
                'error': 'Please provide at least one resume.'
            }), 400
        
        results = []
        selected = []
        not_selected = []
        
        screener = ResumeScreener()
        
        for i, resume_data in enumerate(resumes):
            # Handle both {name, text} and just text formats
            if isinstance(resume_data, dict):
                name = resume_data.get('name', f'Resume {i+1}')
                resume_text = resume_data.get('text', '')
            else:
                name = f'Resume {i+1}'
                resume_text = resume_data
            
            if not resume_text.strip():
                continue
            
            # Fit and analyze
            screener.fit([resume_text], [job_description])
            analysis = screener.get_detailed_analysis(resume_text, job_description)
            
            match_score = analysis['match_score']
            is_selected = match_score >= threshold
            
            result = {
                'name': name,
                'match_score': match_score,
                'is_selected': is_selected,
                'matching_skills': analysis['matching_skills'],
                'missing_skills': analysis['missing_skills']
            }
            
            results.append(result)
            
            if is_selected:
                selected.append(result)
            else:
                not_selected.append(result)
        
        # Sort by match score (highest first)
        results.sort(key=lambda x: x['match_score'], reverse=True)
        selected.sort(key=lambda x: x['match_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'threshold': threshold,
            'total_candidates': len(results),
            'selected_count': len(selected),
            'not_selected_count': len(not_selected),
            'selected': selected,
            'all_results': results,
            'summary': f"{len(selected)} out of {len(results)} candidates selected (≥{threshold}%)"
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/quick-match', methods=['POST'])
def quick_match():
    """
    Quick match endpoint that returns just the percentage match.
    """
    try:
        data = request.get_json()
        
        resume_text = data.get('resume', '').strip()
        job_description = data.get('job_description', '').strip()
        
        if not resume_text or not job_description:
            return jsonify({
                'success': False,
                'error': 'Please provide both resume and job description.'
            }), 400
        
        # Get model and calculate match
        screener = get_model()
        screener.fit([resume_text], [job_description])
        
        match_score = screener.calculate_match_score(resume_text, job_description)
        
        return jsonify({
            'success': True,
            'match_score': match_score,
            'message': f'Your resume matches this role by {match_score}%'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/sample-data', methods=['GET'])
def sample_data():
    """
    Return sample resume and job description for testing.
    """
    sample_resume = """
    JOHN DOE
    Software Engineer
    
    SUMMARY:
    Experienced Software Engineer with 5 years of experience in developing
    web applications and APIs.
    
    SKILLS:
    - Programming: Python, Java, JavaScript
    - Frontend: React, HTML, CSS
    - Backend: Node.js, Django, Flask
    - Database: SQL, PostgreSQL, MongoDB
    - Cloud: AWS, Docker, Kubernetes
    - Tools: Git, Jenkins, JIRA
    
    EXPERIENCE:
    Senior Software Engineer at Tech Corp (2020 - Present)
    - Led development of microservices architecture
    - Implemented RESTful APIs
    - Mentored junior developers
    
    Software Engineer at StartupXYZ (2018 - 2020)
    - Built customer-facing web applications
    - Optimized database queries
    - Implemented CI/CD pipelines
    
    EDUCATION:
    BS Computer Science, University of Technology, 2018
    
    CERTIFICATIONS:
    - AWS Certified Solutions Architect
    - Python Developer Certification
    """
    
    sample_job = """
    Software Engineer Position
    
    About the Role:
    We are looking for a Software Engineer to join our growing team.
    
    Requirements:
    - 3+ years of software development experience
    - Proficiency in Python, Java, or JavaScript
    - Experience with React and Node.js
    - Knowledge of SQL and NoSQL databases
    - Cloud platform experience (AWS preferred)
    - Experience with Docker
    
    Responsibilities:
    - Develop and maintain web applications
    - Design and implement APIs
    - Collaborate with cross-functional teams
    - Write clean, maintainable code
    
    Benefits:
    - Competitive salary
    - Remote work options
    - Health insurance
    - Professional development
    """
    
    return jsonify({
        'success': True,
        'resume': sample_resume,
        'job_description': sample_job
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
