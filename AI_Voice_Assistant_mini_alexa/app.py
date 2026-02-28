"""
Mini Alexa - Web Interface
A Flask-based web interface for the voice assistant
"""

from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import pyttsx3
import wikipedia
import webbrowser
import os
import datetime
import subprocess
from urllib.parse import quote
import pywhatkit

app = Flask(__name__)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Initialize speech recognizer
recognizer = sr.Recognizer()

# App paths
app_paths = {
    'notepad': 'notepad.exe',
    'calculator': 'calc.exe',
    'youtube': None,
    'whatsapp': None,
    'gmail': None,
    'facebook': None,
    'twitter': None,
    'instagram': None,
    'reddit': None,
    'google': None,
    'netflix': None,
}


def speak_text(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/process_command', methods=['POST'])
def process_command():
    """Process a voice command sent from the web interface."""
    data = request.get_json()
    command = data.get('command', '').lower()
    
    response = ""
    action = None
    
    # Greetings
    greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening']
    if any(greet in command for greet in greetings):
        response = "Hello! How can I help you today?"
        
    # Time
    elif 'time' in command:
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        response = f"The current time is {time_str}"
        
    # Date
    elif 'date' in command or 'day' in command:
        now = datetime.datetime.now()
        date_str = now.strftime("%B %d, %Y")
        day_str = now.strftime("%A")
        response = f"Today is {date_str}, which is a {day_str}"
        
    # Search Google
    elif 'search' in command or 'google' in command:
        query = command.replace('search', '').replace('google', '').strip()
        if query:
            search_url = f"https://www.google.com/search?q={quote(query)}"
            webbrowser.open(search_url)
            response = f"Searching for {query}..."
            action = "search"
        
    # Open app
    elif command.startswith('open '):
        app_name = command.replace('open ', '').strip()
        
        # Check web apps
        web_apps = {
            'youtube': 'https://www.youtube.com',
            'whatsapp': 'https://web.whatsapp.com',
            'gmail': 'https://www.gmail.com',
            'facebook': 'https://www.facebook.com',
            'twitter': 'https://www.twitter.com',
            'instagram': 'https://www.instagram.com',
            'reddit': 'https://www.reddit.com',
            'google': 'https://www.google.com',
            'netflix': 'https://www.netflix.com',
        }
        
        if app_name in web_apps:
            webbrowser.open(web_apps[app_name])
            response = f"Opening {app_name}..."
            action = "open_browser"
        elif app_name in app_paths:
            path = app_paths[app_name]
            if path:
                try:
                    os.startfile(path)
                    response = f"Opening {app_name}..."
                    action = "open_app"
                except:
                    response = f"Couldn't open {app_name}."
            else:
                response = f"Opening {app_name} in browser..."
                action = "open_browser"
        else:
            response = f"I couldn't find {app_name}."
            
    # Wikipedia
    elif any(pattern in command for pattern in ['who is', 'what is', 'tell me about']):
        try:
            query = command.replace('who is', '').replace('what is', '').replace('tell me about', '').strip()
            if query:
                summary = wikipedia.summary(query, sentences=2)
                response = summary
        except:
            response = "I couldn't find that information."
            
    # Play YouTube
    elif 'play' in command:
        query = command.replace('play', '').strip()
        if query:
            try:
                pywhatkit.playonyt(query)
                response = f"Playing {query} on YouTube..."
                action = "play_youtube"
            except:
                search_url = f"https://www.youtube.com/results?search_query={quote(query)}"
                webbrowser.open(search_url)
                response = f"Searching YouTube for {query}..."
                action = "search"
                
    # Jokes
    elif 'joke' in command:
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
        ]
        import random
        response = random.choice(jokes)
        
    # Calculator
    elif 'calculate' in command or 'what is' in command:
        expression = command.replace('calculate', '').replace('what is', '').strip()
        expression = expression.replace('x', '*').replace('times', '*')
        expression = expression.replace('plus', '+').replace('minus', '-')
        try:
            result = eval(expression)
            response = f"The answer is {result}"
        except:
            response = "I couldn't calculate that."
            
    # Exit
    elif any(word in command for word in ['exit', 'quit', 'goodbye', 'bye']):
        response = "Goodbye! Have a great day!"
        action = "exit"
        
    # Help
    elif 'help' in command:
        response = """Available commands:
- Ask about time or date
- Search for anything on Google
- Open applications like Notepad, Calculator
- Open websites like YouTube, Gmail, Facebook
- Play videos on YouTube
- Get information from Wikipedia
- Tell jokes
- Calculate math expressions"""
        
    # Default
    else:
        # Try Wikipedia
        try:
            summary = wikipedia.summary(command, sentences=1)
            response = summary
        except:
            # Search Google
            search_url = f"https://www.google.com/search?q={quote(command)}"
            webbrowser.open(search_url)
            response = f"Searching for {command}..."
            action = "search"
    
    return jsonify({
        'response': response,
        'action': action,
        'command': command
    })


@app.route('/speak', methods=['POST'])
def speak():
    """Make the assistant speak the provided text."""
    data = request.get_json()
    text = data.get('text', '')
    
    try:
        speak_text(text)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/listen', methods=['POST'])
def listen():
    """Listen for voice input."""
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            command = recognizer.recognize_google(audio).lower()
            return jsonify({'success': True, 'command': command})
    except sr.UnknownValueError:
        return jsonify({'success': False, 'error': 'Could not understand audio'})
    except sr.RequestError:
        return jsonify({'success': False, 'error': 'Speech service unavailable'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    print("Starting Mini Alexa Web Interface...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)
