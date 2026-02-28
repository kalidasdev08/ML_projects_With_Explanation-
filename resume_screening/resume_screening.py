"""
Resume Screening AI - Core NLP Module
Uses TF-IDF embeddings, cosine similarity, and keyword extraction
to match candidate resumes with job descriptions.
"""

import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle
import os


class ResumeScreener:
    """
    AI-powered resume screening tool that matches resumes with job descriptions
    using NLP techniques: TF-IDF embeddings, cosine similarity, and keyword extraction.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Use unigrams and bigrams
            max_features=5000,
            min_df=1,
            max_df=0.95
        )
        self.resume_vectors = None
        self.job_desc_vectors = None
        self.resume_texts = []
        self.job_desc_texts = []
        self.feature_names = []
        
    def preprocess_text(self, text):
        """
        Preprocess text by cleaning and normalizing.
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\-\+\.]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, text, top_n=10):
        """
        Extract top keywords from text using TF-IDF.
        """
        if not text:
            return []
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Create a mini vectorizer for keyword extraction
        keyword_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 1),
            max_features=1000
        )
        
        try:
            # Fit on single document (needs at least 2 documents for some versions)
            keyword_vectorizer.fit([processed_text, " "])
            tfidf_matrix = keyword_vectorizer.transform([processed_text])
            
            # Get feature names and their scores
            feature_names = keyword_vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top N keywords
            top_indices = scores.argsort()[-top_n:][::-1]
            keywords = [(feature_names[i], scores[i]) for i in top_indices if scores[i] > 0]
            
            return keywords
        except:
            return []
    
    def extract_skills(self, text):
        """
        Extract technical skills from resume/job description.
        """
        # Common technical skills database
        technical_skills = [
            # Programming Languages
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'go', 'rust',
            'typescript', 'php', 'swift', 'kotlin', 'scala', 'r', 'matlab',
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue', 'node.js', 'express',
            'django', 'flask', 'spring', 'bootstrap', 'jquery', 'rest api',
            'graphql', 'ajax', 'sass', 'less',
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'oracle',
            'sqlite', 'elasticsearch', 'cassandra', 'firebase',
            # Data Science & ML
            'machine learning', 'deep learning', 'tensorflow', 'pytorch',
            'keras', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            'data analysis', 'data visualization', 'nlp', 'computer vision',
            'artificial intelligence', 'ai', 'ml',
            # Cloud & DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
            'terraform', 'ansible', 'ci/cd', 'devops', 'linux', 'git',
            # Big Data
            'hadoop', 'spark', 'hive', 'kafka', 'pig', 'mapreduce',
            # Tools
            'jira', 'confluence', 'slack', 'excel', 'powerpoint', 'word',
            'tableau', 'power bi', 'looker',
            # Soft Skills
            'leadership', 'communication', 'teamwork', 'problem-solving',
            'analytical', 'project management', 'agile', 'scrum'
        ]
        
        text_lower = text.lower()
        found_skills = []
        
        for skill in technical_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        return found_skills
    
    def fit(self, resumes, job_descriptions):
        """
        Fit the model on resumes and job descriptions.
        """
        # Preprocess texts
        self.resume_texts = [self.preprocess_text(r) for r in resumes]
        self.job_desc_texts = [self.preprocess_text(j) for j in job_descriptions]
        
        # Add some placeholder texts to help TF-IDF learn proper IDF values
        # This ensures we get meaningful similarity scores
        placeholder_texts = [
            "software engineer developer programmer code programming",
            "python java javascript react node sql database cloud",
            "machine learning data science analysis visualization",
            "project management team work communication skills",
            "education degree computer science engineering"
        ]
        
        # Combine all texts for vectorizer fitting
        all_texts = self.resume_texts + self.job_desc_texts + placeholder_texts
        
        # Fit vectorizer
        self.vectorizer.fit(all_texts)
        
        # Transform texts
        self.resume_vectors = self.vectorizer.transform(self.resume_texts)
        self.job_desc_vectors = self.vectorizer.transform(self.job_desc_texts)
        
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        
        return self
    
    def calculate_match_score(self, resume_text, job_desc_text):
        """
        Calculate match score between a resume and job description.
        Returns a percentage match score (0-100).
        """
        # Preprocess texts
        resume_processed = self.preprocess_text(resume_text)
        job_desc_processed = self.preprocess_text(job_desc_text)
        
        # Transform texts
        resume_vec = self.vectorizer.transform([resume_processed])
        job_desc_vec = self.vectorizer.transform([job_desc_processed])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_vec, job_desc_vec)[0][0]
        
        # Also check word overlap (more lenient filtering)
        resume_words = set(resume_processed.split())
        job_words = set(job_desc_processed.split())
        
        # Only remove very basic stop words, keep important keywords
        minimal_stop_words = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for',
                             'of', 'with', 'by', 'from', 'is', 'was', 'are', 'were', 'been',
                             'be', 'have', 'has', 'had', 'do', 'does', 'did'}
        
        resume_words = resume_words - minimal_stop_words
        job_words = job_words - minimal_stop_words
        
        # Calculate overlap ratio
        if len(job_words) > 0:
            overlap = len(resume_words & job_words)
            overlap_ratio = overlap / len(job_words)
            
            # Use the maximum of cosine similarity and word overlap
            # This ensures we get a meaningful score even with limited vocabulary
            match_score = max(similarity, overlap_ratio) * 100
        else:
            match_score = similarity * 100
        
        # Ensure reasonable minimum based on skills match
        resume_skills = set(self.extract_skills(resume_text))
        job_skills = set(self.extract_skills(job_desc_text))
        
        if len(job_skills) > 0:
            skill_match = len(resume_skills & job_skills) / len(job_skills)
            # Use skills match if it's higher
            match_score = max(match_score, skill_match * 100)
        
        # Cap at 100
        match_score = min(round(match_score, 2), 100)
        
        return match_score
    
    def get_detailed_analysis(self, resume_text, job_desc_text):
        """
        Get detailed analysis including keywords, skills match, and recommendations.
        """
        # Get match score
        match_score = self.calculate_match_score(resume_text, job_desc_text)
        
        # Extract resume keywords
        resume_keywords = self.extract_keywords(resume_text, top_n=15)
        
        # Extract job description keywords
        job_keywords = self.extract_keywords(job_desc_text, top_n=15)
        
        # Extract skills
        resume_skills = self.extract_skills(resume_text)
        job_skills = self.extract_skills(job_desc_text)
        
        # Find matching skills
        matching_skills = set(resume_skills) & set(job_skills)
        missing_skills = set(job_skills) - set(resume_skills)
        
        # Find matching keywords
        resume_keyword_set = set([k[0] for k in resume_keywords])
        job_keyword_set = set([k[0] for k in job_keywords])
        matching_keywords = resume_keyword_set & job_keyword_set
        
        # Calculate keyword match percentage
        if len(job_keyword_set) > 0:
            keyword_match_pct = round(len(matching_keywords) / len(job_keyword_set) * 100, 2)
        else:
            keyword_match_pct = 0
        
        # Calculate skill match percentage
        if len(job_skills) > 0:
            skill_match_pct = round(len(matching_skills) / len(job_skills) * 100, 2)
        else:
            skill_match_pct = 0
        
        # Generate recommendations
        recommendations = []
        
        if match_score < 50:
            recommendations.append("Consider tailoring your resume more closely to the job requirements.")
        if match_score < 70:
            recommendations.append("Add more keywords from the job description to your resume.")
        
        if missing_skills:
            recommendations.append(f"Consider highlighting or acquiring skills: {', '.join(list(missing_skills)[:5])}")
        
        if len(matching_skills) < len(job_skills):
            recommendations.append("Ensure your existing relevant skills are prominently displayed.")
        
        return {
            'match_score': match_score,
            'keyword_match_percentage': keyword_match_pct,
            'skill_match_percentage': skill_match_pct,
            'resume_keywords': [k[0] for k in resume_keywords[:10]],
            'job_keywords': [k[0] for k in job_keywords[:10]],
            'matching_keywords': list(matching_keywords),
            'resume_skills': resume_skills,
            'job_skills': job_skills,
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'recommendations': recommendations
        }
    
    def save_model(self, filepath):
        """
        Save the model to a pickle file.
        """
        model_data = {
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load_model(cls, filepath):
        """
        Load a saved model from a pickle file.
        """
        instance = cls()
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        instance.vectorizer = model_data['vectorizer']
        instance.feature_names = model_data['feature_names']
        return instance


def match_resume_to_job(resume_text, job_description, model=None):
    """
    Convenience function to match a resume to a job description.
    Returns the match score as a percentage.
    """
    if model is None:
        # Create a new model and fit on the two texts
        model = ResumeScreener()
        model.fit([resume_text], [job_description])
    
    return model.calculate_match_score(resume_text, job_description)


def get_full_analysis(resume_text, job_description):
    """
    Get full analysis of resume-job match.
    """
    model = ResumeScreener()
    model.fit([resume_text], [job_description])
    
    return model.get_detailed_analysis(resume_text, job_description)


if __name__ == "__main__":
    # Example usage
    sample_resume = """
    JOHN DOE
    Software Engineer
    
    SKILLS:
    - Python, Java, JavaScript
    - React, Node.js, Django
    - SQL, PostgreSQL, MongoDB
    - AWS, Docker, Git
    - Machine Learning, TensorFlow
    
    EXPERIENCE:
    - Senior Software Engineer at Tech Corp (2020-Present)
    - Developed web applications using Python and React
    - Implemented REST APIs
    - Worked with AWS cloud services
    
    EDUCATION:
    - BS Computer Science, University of Tech
    
    PROJECTS:
    - Built a machine learning model for prediction
    - Created a REST API for data management
    """
    
    sample_job = """
    We are looking for a Software Engineer with:
    
    Required Skills:
    - Python, JavaScript
    - React, Node.js
    - SQL, MongoDB
    - AWS experience
    - Machine Learning knowledge
    
    Responsibilities:
    - Develop web applications
    - Implement APIs
    - Work with cloud services
    """
    
    # Test the resume screener
    screener = ResumeScreener()
    screener.fit([sample_resume], [sample_job])
    
    analysis = screener.get_detailed_analysis(sample_resume, sample_job)
    
    print(f"\n{'='*60}")
    print(f"RESUME SCREENING RESULTS")
    print(f"{'='*60}")
    print(f"\n🎯 Overall Match Score: {analysis['match_score']}%")
    print(f"📝 Keyword Match: {analysis['keyword_match_percentage']}%")
    print(f"🛠️ Skill Match: {analysis['skill_match_percentage']}%")
    print(f"\n✅ Matching Skills: {', '.join(analysis['matching_skills']) if analysis['matching_skills'] else 'None'}")
    print(f"❌ Missing Skills: {', '.join(analysis['missing_skills']) if analysis['missing_skills'] else 'None'}")
    print(f"\n📋 Recommendations:")
    for rec in analysis['recommendations']:
        print(f"   • {rec}")
    print(f"\n{'='*60}")
