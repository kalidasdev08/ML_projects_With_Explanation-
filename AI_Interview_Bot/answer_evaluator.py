"""
Answer Evaluation Module for AI Interview Bot
Uses Google Gemini API to generate questions and evaluate answers
"""

import json
import os
import re
from typing import Dict, List, Tuple, Optional
import google.generativeai as genai

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')


class InterviewBot:
    """AI Interview Bot that generates questions and evaluates answers using Gemini"""
    
    def __init__(self, api_key: str = None):
        """Initialize the interview bot"""
        self.api_key = api_key or GEMINI_API_KEY
        self.model = None
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            print("Warning: No Gemini API key provided.")
    
    def generate_question(self, category: str = None, job_role: str = None, difficulty: str = "medium") -> Dict:
        """
        Generate a unique interview question using Gemini API
        
        Args:
            category: Category of question (technical, behavioral, situational)
            job_role: Target job role for personalized questions
            difficulty: Difficulty level (easy, medium, hard)
            
        Returns:
            Dictionary containing the generated question and metadata
        """
        if not self.model:
            return self._fallback_question(category)
        
        # Build prompt for question generation
        category_info = {
            "technical": "technical skills, coding, problem-solving, system design",
            "behavioral": "past experiences, teamwork, conflict resolution, leadership",
            "situational": "hypothetical scenarios, decision-making, problem-solving",
            "general": "general interview questions, self-awareness, career goals"
        }
        
        category_text = category_info.get(category, "general interview questions") if category else "general interview questions"
        role_context = f" for a {job_role} position" if job_role else ""
        
        prompt = f"""Generate a unique interview question{role_context} focused on {category_text}.

The question should be at {difficulty} difficulty level.

Respond in JSON format with exactly this structure:
{{
    "id": "unique_id",
    "question": "The actual question text",
    "category": "{category or 'general'}",
    "type": "open-ended" or "scenario-based" or "behavioral",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "expected_depth": "What kind of answer depth you expect (brief/detailed)",
    "tips": ["tip1", "tip2"]
}}

Make sure the question is original and not repetitive. Generate a unique question each time.
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                question_data = json.loads(json_match.group())
                return question_data
            else:
                return self._fallback_question(category)
                
        except Exception as e:
            print(f"Error generating question: {e}")
            return self._fallback_question(category)
    
    def _fallback_question(self, category: str = None) -> Dict:
        """Fallback question generation without API"""
        import random
        
        fallback_questions = {
            "technical": [
                "Describe a complex technical problem you solved. What was your approach?",
                "Explain a technical concept you're passionate about.",
                "How do you stay updated with the latest technology trends?"
            ],
            "behavioral": [
                "Tell me about a challenging situation you faced at work and how you handled it.",
                "Describe a time when you had to work with a difficult team member.",
                "What is your greatest strength and how does it help you in your work?"
            ],
            "situational": [
                "How would you handle a project with unrealistic deadlines?",
                "What would you do if you disagreed with your manager's decision?",
                "How would you prioritize multiple urgent tasks?"
            ],
            "general": [
                "Tell me about yourself and why you're a good fit for this role.",
                "What are your career goals for the next 5 years?",
                "Why do you want to work for this company?"
            ]
        }
        
        questions = fallback_questions.get(category, fallback_questions["general"])
        question_text = random.choice(questions)
        
        return {
            "id": f"fallback_{category}_{random.randint(1000, 9999)}",
            "question": question_text,
            "category": category or "general",
            "type": "open-ended",
            "keywords": ["experience", "skills", "role"],
            "expected_depth": "detailed",
            "tips": ["Be specific", "Use examples"]
        }
    
    def generate_multiple_questions(self, count: int = 5, category: str = None, job_role: str = None) -> List[Dict]:
        """
        Generate multiple unique interview questions
        
        Args:
            count: Number of questions to generate
            category: Category of questions
            job_role: Target job role
            
        Returns:
            List of question dictionaries
        """
        questions = []
        categories = [category] if category else ["technical", "behavioral", "situational", "general"]
        
        for i in range(count):
            cat = categories[i % len(categories)]
            question = self.generate_question(category=cat, job_role=job_role, difficulty="medium")
            questions.append(question)
        
        return questions
    
    def evaluate_answer(self, answer: str, question: str, question_data: Dict = None) -> Dict:
        """
        Evaluate answer using Gemini API
        
        Args:
            answer: The user's answer text
            question: The interview question
            question_data: Additional question metadata
            
        Returns:
            Dictionary containing scores and detailed feedback
        """
        if not self.model:
            return self._fallback_evaluation(answer, question)
        
        keywords = question_data.get('keywords', []) if question_data else []
        keywords_text = ", ".join(keywords) if keywords else "relevant experience, skills, examples"
        
        prompt = f"""You are an expert interview coach evaluating a candidate's answer to an interview question.

Question: {question}

Candidate's Answer: {answer}

Evaluate this answer based on the following criteria (score each from 0-100):
1. Clarity: How clearly is the answer communicated? Is it well-structured?
2. Relevance: How relevant is the answer to the question? Does it address what's being asked?
3. Depth: How detailed and thorough is the response? Does it show depth of experience?
4. Specificity: Does the answer include specific examples, numbers, or concrete details?

Also provide:
- Overall Score (0-100): A weighted average (Clarity 25%, Relevance 30%, Depth 25%, Specificity 20%)
- Strengths: What did the candidate do well? (list 2-3 specific things)
- Areas for Improvement: What could be improved? (list 2-3 specific suggestions)
- Detailed Feedback: 2-3 sentence actionable feedback

Respond in JSON format:
{{
    "clarity": <score>,
    "relevance": <score>,
    "depth": <score>,
    "specificity": <score>,
    "overall_score": <score>,
    "strengths": ["strength1", "strength2", "strength3"],
    "areas_for_improvement": ["area1", "area2", "area3"],
    "detailed_feedback": "Your 2-3 sentence feedback here"
}}
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                result = json.loads(json_match.group())
                result['word_count'] = len(answer.split())
                return result
            else:
                return self._fallback_evaluation(answer, question)
                
        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return self._fallback_evaluation(answer, question)
    
    def _fallback_evaluation(self, answer: str, question: str) -> Dict:
        """Fallback evaluation using basic analysis when API fails"""
        if not answer or not answer.strip():
            return {
                "clarity": 0,
                "relevance": 0,
                "depth": 0,
                "specificity": 0,
                "overall_score": 0,
                "strengths": [],
                "areas_for_improvement": ["No answer provided"],
                "detailed_feedback": "Please provide an answer to receive evaluation."
            }
        
        words = answer.split()
        word_count = len(words)
        
        # Basic pattern analysis
        has_examples = bool(re.search(r'\b(I worked|I led|I managed|I created|I developed|I achieved)\b', answer.lower()))
        has_numbers = bool(re.search(r'\b(\d+|percent|percentage|increased|decreased)\b', answer.lower()))
        has_conclusions = bool(re.search(r'\b(therefore|so|finally|in conclusion|result)\b', answer.lower()))
        
        clarity = min(100, max(40, 60 + (20 if word_count > 20 else -10) + (10 if has_conclusions else 0)))
        relevance = min(100, max(50, 70 + (15 if word_count > 30 else -15)))
        depth = min(100, max(30, 50 + (20 if word_count > 50 else 0) + (20 if has_examples else 0)))
        specificity = min(100, max(20, 40 + (25 if has_examples else 0) + (25 if has_numbers else 0)))
        
        overall = int((clarity * 0.25) + (relevance * 0.30) + (depth * 0.25) + (specificity * 0.20))
        
        strengths = []
        areas = []
        
        if word_count > 30:
            strengths.append("Good answer length")
        else:
            areas.append("Consider providing more detail")
        
        if has_examples:
            strengths.append("Good use of specific examples")
        else:
            areas.append("Add specific examples from your experience")
        
        if has_numbers:
            strengths.append("Good use of quantifiable details")
        
        return {
            "clarity": clarity,
            "relevance": relevance,
            "depth": depth,
            "specificity": specificity,
            "overall_score": overall,
            "strengths": strengths if strengths else ["Attempted to answer"],
            "areas_for_improvement": areas if areas else ["Keep practicing"],
            "detailed_feedback": " ".join(areas) if areas else "Good effort in your response."
        }
    
    def generate_follow_up(self, question: str, answer: str) -> str:
        """Generate a follow-up question based on the answer"""
        if not self.model:
            return "Can you provide more specific details about that?"
        
        prompt = f"""Based on this interview question and answer, generate ONE concise follow-up question to probe deeper.

Original Question: {question}

Candidate's Answer: {answer}

Provide only the follow-up question (max 2 sentences). Make it specific and relevant to what the candidate mentioned.
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except:
            return "Can you tell me more about that?"
    
    def generate_questions_for_role(self, job_role: str, count: int = 5) -> List[Dict]:
        """
        Generate personalized questions for a specific job role
        
        Args:
            job_role: The target job role
            count: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        if not self.model:
            return [self._fallback_question() for _ in range(count)]
        
        prompt = f"""Generate {count} unique interview questions for a {job_role} position.

Include a mix of:
- 2 technical questions related to {job_role}
- 2 behavioral questions about teamwork and challenges
- 1 situational question about problem-solving

Each question should be original and specific to the {job_role} role.

Respond in JSON format as an array of question objects:
[
    {{
        "id": "q1",
        "question": "question text here",
        "category": "technical/behavioral/situational",
        "type": "open-ended/scenario-based",
        "keywords": ["keyword1", "keyword2"],
        "expected_depth": "brief/detailed"
    }}
]
"""
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Find JSON array in response
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                questions = json.loads(json_match.group())
                return questions
            else:
                return self.generate_multiple_questions(count=count, job_role=job_role)
                
        except Exception as e:
            print(f"Error generating role-specific questions: {e}")
            return self.generate_multiple_questions(count=count)


# For testing
if __name__ == "__main__":
    import os
    
    api_key = os.environ.get('GEMINI_API_KEY')
    bot = InterviewBot(api_key=api_key)
    
    if bot.model:
        print("Testing question generation...")
        question = bot.generate_question(category="technical", job_role="Software Engineer")
        print(f"\nGenerated Question:\n{json.dumps(question, indent=2)}")
        
        print("\n\nTesting answer evaluation...")
        test_answer = "I worked on a project where I had to develop a web application using Python and Flask. It was challenging because I had to learn new technologies quickly, but I managed to complete it on time by dedicating extra hours to learning."
        
        evaluation = bot.evaluate_answer(test_answer, question['question'], question)
        print(f"\nEvaluation Result:\n{json.dumps(evaluation, indent=2)}")
    else:
        print("No API key provided. Please set GEMINI_API_KEY environment variable.")
