"""
Resume Screening AI - Model Training Module
Trains the NLP model for resume-job description matching.
"""

import os
import pickle
from resume_screening import ResumeScreener


def train_model():
    """
    Train the resume screening model using sample data.
    """
    # Sample resumes (in real scenario, load from files/database)
    sample_resumes = [
        """
        JOHN SMITH
        Software Engineer
        
        Skills: Python, Java, JavaScript, React, Node.js, SQL, MongoDB, AWS, Docker, Git
        
        Experience:
        - Software Engineer at Tech Solutions (2021-Present)
        - Developed full-stack web applications
        - Implemented REST APIs
        - Worked with cloud services
        
        Education:
        - BS Computer Science
        
        Projects:
        - E-commerce platform using React and Django
        - Data analytics dashboard
        """,
        
        """
        JANE DOE
        Data Scientist
        
        Skills: Python, R, Machine Learning, TensorFlow, Pandas, SQL, Tableau
        
        Experience:
        - Data Scientist at Data Corp (2020-Present)
        - Built predictive models
        - Performed data analysis
        - Created visualizations
        
        Education:
        - MS Data Science
        """,
        
        """
        ALEX JOHNSON
        Full Stack Developer
        
        Skills: JavaScript, TypeScript, React, Angular, Node.js, Express, MongoDB, PostgreSQL
        
        Experience:
        - Full Stack Developer at WebTech (2019-Present)
        - Developed web applications
        - Implemented microservices
        - Managed databases
        
        Education:
        - BS Software Engineering
        """,
        
        """
        SARAH WILLIAMS
        DevOps Engineer
        
        Skills: AWS, Azure, Docker, Kubernetes, Jenkins, Terraform, Python, Linux, Bash
        
        Experience:
        - DevOps Engineer at CloudTech (2020-Present)
        - Set up CI/CD pipelines
        - Managed cloud infrastructure
        - Implemented automation
        
        Education:
        - BS Information Technology
        """,
        
        """
        MICHAEL BROWN
        Backend Developer
        
        Skills: Python, Django, Flask, PostgreSQL, Redis, RabbitMQ, Docker
        
        Experience:
        - Backend Developer at API Solutions (2021-Present)
        - Built REST APIs
        - Optimized database queries
        - Implemented caching
        
        Education:
        - BS Computer Engineering
        """
    ]
    
    # Sample job descriptions
    sample_jobs = [
        """
        Software Engineer Position
        
        Requirements:
        - Python, Java, JavaScript
        - React, Node.js
        - SQL, MongoDB
        - AWS experience
        - Full-stack development experience
        
        Responsibilities:
        - Develop web applications
        - Implement APIs
        - Work with databases
        """,
        
        """
        Data Scientist Role
        
        Requirements:
        - Python, R
        - Machine Learning
        - TensorFlow, PyTorch
        - SQL, Pandas
        - Data visualization
        
        Responsibilities:
        - Build ML models
        - Analyze data
        - Create reports
        """,
        
        """
        Full Stack Developer
        
        Requirements:
        - JavaScript, TypeScript
        - React, Angular
        - Node.js, Express
        - MongoDB, PostgreSQL
        
        Responsibilities:
        - Develop web applications
        - Create APIs
        - Manage databases
        """,
        
        """
        DevOps Engineer
        
        Requirements:
        - AWS, Azure
        - Docker, Kubernetes
        - Jenkins, CI/CD
        - Terraform
        - Python, Linux
        
        Responsibilities:
        - Manage infrastructure
        - Set up pipelines
        - Implement automation
        """,
        
        """
        Backend Developer
        
        Requirements:
        - Python
        - Django, Flask
        - PostgreSQL
        - Redis
        - Docker
        
        Responsibilities:
        - Build APIs
        - Optimize performance
        - Manage databases
        """
    ]
    
    # Train the model
    print("Training Resume Screening Model...")
    model = ResumeScreener()
    model.fit(sample_resumes, sample_jobs)
    
    # Save the model
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, 'resume_screener.pkl')
    model.save_model(model_path)
    
    print(f"Model saved to: {model_path}")
    
    # Test the model
    test_resume = sample_resumes[0]
    test_job = sample_jobs[0]
    
    match_score = model.calculate_match_score(test_resume, test_job)
    print(f"\nTest Match Score: {match_score}%")
    
    # Get detailed analysis
    analysis = model.get_detailed_analysis(test_resume, test_job)
    print(f"\nDetailed Analysis:")
    print(f"  Keyword Match: {analysis['keyword_match_percentage']}%")
    print(f"  Skill Match: {analysis['skill_match_percentage']}%")
    print(f"  Matching Skills: {analysis['matching_skills']}")
    print(f"  Missing Skills: {analysis['missing_skills']}")
    
    return model


def load_trained_model():
    """
    Load a pre-trained model.
    """
    model_path = 'models/resume_screener.pkl'
    if os.path.exists(model_path):
        return ResumeScreener.load_model(model_path)
    else:
        print("No pre-trained model found. Training new model...")
        return train_model()


if __name__ == "__main__":
    model = train_model()
    print("\n✅ Model training complete!")
