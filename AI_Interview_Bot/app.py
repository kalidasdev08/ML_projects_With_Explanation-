"""
AI Interview Bot - Flask Web Application
A chatbot that generates unique interview questions using Gemini AI and evaluates answers
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
from answer_evaluator import InterviewBot

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'ai-interview-bot-secret-key-2024')

# Initialize the interview bot
def get_bot():
    """Get or create the InterviewBot instance"""
    api_key = os.environ.get('GEMINI_API_KEY')
    return InterviewBot(api_key=api_key)


@app.route('/')
def index():
    """Home page"""
    if 'interview' not in session:
        session['interview'] = {
            'current_question_index': 0,
            'questions_asked': [],
            'answers': [],
            'scores': [],
            'category': None,
            'job_role': None,
            'started': True
        }
    
    return render_template('index.html')


@app.route('/start-interview', methods=['POST'])
def start_interview():
    """Start a new interview session with AI-generated questions"""
    data = request.get_json() or {}
    category = data.get('category', 'all')
    job_role = data.get('job_role', '')
    
    bot = get_bot()
    
    # Generate unique questions using AI
    if job_role and job_role.strip():
        # Generate role-specific questions
        questions = bot.generate_questions_for_role(job_role.strip(), count=5)
    elif category == 'all':
        # Generate mixed questions from all categories
        questions = bot.generate_multiple_questions(count=5)
    else:
        # Generate questions for specific category
        questions = bot.generate_multiple_questions(count=5, category=category)
    
    # Store in session
    session['interview'] = {
        'questions': questions,
        'current_index': 0,
        'answers': [],
        'scores': [],
        'category': category,
        'job_role': job_role,
        'started': True
    }
    
    first_question = questions[0] if questions else None
    
    return jsonify({
        'success': True,
        'question': first_question,
        'total_questions': len(questions),
        'current_question': 1
    })


@app.route('/get-question', methods=['GET'])
def get_question():
    """Get the current question"""
    interview = session.get('interview', {})
    questions = interview.get('questions', [])
    current_index = interview.get('current_index', 0)
    
    if current_index < len(questions):
        question = questions[current_index]
        return jsonify({
            'success': True,
            'question': question,
            'current_question': current_index + 1,
            'total_questions': len(questions)
        })
    
    return jsonify({
        'success': False,
        'message': 'No more questions'
    })


@app.route('/submit-answer', methods=['POST'])
def submit_answer():
    """Submit an answer and get AI evaluation"""
    data = request.get_json()
    answer = data.get('answer', '')
    question_id = data.get('question_id', '')
    
    interview = session.get('interview', {})
    questions = interview.get('questions', [])
    current_index = interview.get('current_index', 0)
    
    # Find the current question
    current_question = None
    question_text = ""
    for q in questions:
        if q.get('id') == question_id:
            current_question = q
            question_text = q.get('question', '')
            break
    
    # Also try by index
    if not current_question and current_index < len(questions):
        current_question = questions[current_index]
        question_text = current_question.get('question', '')
    
    if not question_text:
        return jsonify({
            'success': False,
            'message': 'Question not found'
        })
    
    # Evaluate the answer using AI
    bot = get_bot()
    evaluation = bot.evaluate_answer(answer, question_text, current_question)
    
    # Store the answer and score
    if 'answers' not in session.get('interview', {}):
        session['interview'] = interview
        interview['answers'] = []
        interview['scores'] = []
    
    interview['answers'].append({
        'question': question_text,
        'answer': answer,
        'question_id': question_id
    })
    
    interview['scores'].append(evaluation)
    
    # Generate follow-up question
    follow_up = bot.generate_follow_up(question_text, answer)
    
    session['interview'] = interview
    
    return jsonify({
        'success': True,
        'evaluation': evaluation,
        'follow_up': follow_up
    })


@app.route('/next-question', methods=['POST'])
def next_question():
    """Move to the next question"""
    interview = session.get('interview', {})
    current_index = interview.get('current_index', 0)
    questions = interview.get('questions', [])
    
    # Move to next question
    interview['current_index'] = current_index + 1
    session['interview'] = interview
    
    if current_index + 1 < len(questions):
        next_q = questions[current_index + 1]
        return jsonify({
            'success': True,
            'question': next_q,
            'current_question': current_index + 2,
            'total_questions': len(questions)
        })
    
    return jsonify({
        'success': False,
        'message': 'Interview complete'
    })


@app.route('/get-results', methods=['GET'])
def get_results():
    """Get the final results of the interview"""
    interview = session.get('interview', {})
    scores = interview.get('scores', [])
    answers = interview.get('answers', [])
    
    if not scores:
        return jsonify({
            'success': False,
            'message': 'No results available'
        })
    
    # Calculate average scores
    avg_overall = sum(s.get('overall_score', 0) for s in scores) / len(scores) if scores else 0
    avg_clarity = sum(s.get('clarity', 0) for s in scores) / len(scores) if scores else 0
    avg_relevance = sum(s.get('relevance', 0) for s in scores) / len(scores) if scores else 0
    avg_depth = sum(s.get('depth', 0) for s in scores) / len(scores) if scores else 0
    avg_specificity = sum(s.get('specificity', 0) for s in scores) / len(scores) if scores else 0
    
    # Determine overall rating
    if avg_overall >= 80:
        rating = "Excellent"
        description = "You're very well prepared for this interview!"
    elif avg_overall >= 60:
        rating = "Good"
        description = "You have a solid foundation. Keep practicing to improve."
    elif avg_overall >= 40:
        rating = "Fair"
        description = "There's room for improvement. Review the feedback."
    else:
        rating = "Needs Work"
        description = "Keep practicing! Consider preparing more examples."
    
    return jsonify({
        'success': True,
        'results': {
            'total_questions': len(scores),
            'average_scores': {
                'overall': round(avg_overall, 1),
                'clarity': round(avg_clarity, 1),
                'relevance': round(avg_relevance, 1),
                'depth': round(avg_depth, 1),
                'specificity': round(avg_specificity, 1)
            },
            'rating': rating,
            'description': description,
            'answers': [
                {
                    'question': a.get('question'),
                    'answer': a.get('answer'),
                    'score': s.get('overall_score'),
                    'feedback': s.get('detailed_feedback', '')
                }
                for a, s in zip(answers, scores)
            ]
        }
    })


@app.route('/reset-interview', methods=['POST'])
def reset_interview():
    """Reset the interview session"""
    session.pop('interview', None)
    return jsonify({'success': True})


@app.route('/get-categories', methods=['GET'])
def get_categories():
    """Get available question categories"""
    return jsonify({
        'success': True,
        'categories': [
            {'id': 'technical', 'name': '💻 Technical Questions'},
            {'id': 'behavioral', 'name': '🎯 Behavioral Questions'},
            {'id': 'situational', 'name': '⚡ Situational Questions'},
            {'id': 'general', 'name': '📋 General Questions'}
        ]
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    print(f"Starting AI Interview Bot on port {port}")
    print(f"Gemini API configured: {bool(os.environ.get('GEMINI_API_KEY'))}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
