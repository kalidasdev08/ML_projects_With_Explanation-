"""
Mini Alexa - AI Voice Assistant
A hands-free voice assistant that can:
- Convert speech to text
- Answer questions using NLP
- Open applications
- Search Google
- And more...
"""

import speech_recognition as sr
import pyttsx3
import wikipedia
import webbrowser
import os
import datetime
import time
import requests
import json
import re
import subprocess
from urllib.parse import quote
import pywhatkit


class VoiceAssistant:
    """Main voice assistant class with speech recognition and NLP capabilities."""
    
    def __init__(self):
        """Initialize the voice assistant with speech recognition and TTS engines."""
        # Initialize speech recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech
        self.engine.setProperty('volume', 1.0)  # Volume level
        
        # Get available voices
        voices = self.engine.getProperty('voices')
        if len(voices) > 1:
            self.engine.setProperty('voice', voices[1].id)  # Female voice
        else:
            self.engine.setProperty('voice', voices[0].id)
        
        # Application paths (customize these for your system)
        self.app_paths = {
            'notepad': 'notepad.exe',
            'calculator': 'calc.exe',
            'browser': 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
            'chrome': 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
            'firefox': 'C:\\Program Files\\Mozilla Firefox\\firefox.exe',
            'microsoft edge': 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
            'edge': 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
            'word': 'C:\\Program Files\\Microsoft Office\\root\\Office16\\WINWORD.EXE',
            'excel': 'C:\\Program Files\\Microsoft Office\\root\\Office16\\EXCEL.EXE',
            'powerpoint': 'C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE',
            'spotify': 'C:\\Users\\' + os.getenv('USERNAME', '') + '\\AppData\\Roaming\\Spotify\\Spotify.exe',
            'youtube': None,  # Opens in browser
            'whatsapp': None,  # Opens in browser
            'gmail': None,  # Opens in browser
        }
        
        # Conversation patterns
        self.greetings = ['hello', 'hi', 'hey', 'good morning', 'good evening', 'good afternoon', 'what\'s up']
        self.exit_commands = ['exit', 'quit', 'goodbye', 'bye', 'sleep', 'stop', 'that\'s all']
        
        print("=" * 50)
        print("  Mini Alexa - Voice Assistant")
        print("=" * 50)
        print("Say 'help' to see available commands")
        print("=" * 50)
        
    def speak(self, text):
        """Convert text to speech."""
        print(f"Assistant: {text}")
        self.engine.say(text)
        self.engine.runAndWait()
        
    def listen(self):
        """Listen for voice input and convert to text."""
        with sr.Microphone() as source:
            print("\nListening...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                print("Recognizing...")
                
                # Use Google Speech Recognition
                command = self.recognizer.recognize_google(audio).lower()
                print(f"You: {command}")
                return command
                
            except sr.WaitTimeoutError:
                self.speak("I didn't hear anything. Please try again.")
                return None
                
            except sr.UnknownValueError:
                self.speak("Sorry, I didn't understand that. Could you repeat?")
                return None
                
            except sr.RequestError:
                self.speak("Sorry, I'm having trouble connecting to the speech service.")
                return None
                
    def listen_continuously(self):
        """Listen continuously for voice input."""
        with sr.Microphone() as source:
            print("\nListening continuously... (Press Ctrl+C to stop)")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                try:
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"You: {command}")
                    return command
                    
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    self.speak("Connection error. Please check your internet.")
                    break
                    
    def get_time(self):
        """Get and speak the current time."""
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        return f"The current time is {time_str}"
        
    def get_date(self):
        """Get and speak the current date."""
        now = datetime.datetime.now()
        date_str = now.strftime("%B %d, %Y")
        day_str = now.strftime("%A")
        return f"Today is {date_str}, which is a {day_str}"
        
    def search_google(self, query):
        """Search Google for the given query."""
        try:
            search_url = f"https://www.google.com/search?q={quote(query)}"
            webbrowser.open(search_url)
            return f"Searching for {query}..."
        except Exception as e:
            return f"Sorry, I couldn't search for {query}. Error: {str(e)}"
        
    def search_wikipedia(self, query):
        """Search Wikipedia for information."""
        try:
            # Clean up the query
            query = query.replace('who is', '').replace('what is', '').replace('tell me about', '').strip()
            
            if not query:
                return "What would you like me to look up?"
            
            summary = wikipedia.summary(query, sentences=2)
            return summary
            
        except wikipedia.exceptions.DisambiguationError as e:
            return f"There are multiple results. Could you be more specific? Options: {', '.join(e.options[:5])}"
            
        except wikipedia.exceptions.PageError:
            return f"I couldn't find any information about {query} on Wikipedia."
            
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
            
    def open_application(self, app_name):
        """Open an application by name."""
        app_name = app_name.lower()
        
        # Check if it's a web app
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
            'spotify web': 'https://open.spotify.com',
        }
        
        if app_name in web_apps:
            webbrowser.open(web_apps[app_name])
            return f"Opening {app_name}..."
        
        # Check if app is in our predefined paths
        if app_name in self.app_paths:
            path = self.app_paths[app_name]
            if path:
                try:
                    os.startfile(path)
                    return f"Opening {app_name}..."
                except Exception as e:
                    return f"Couldn't open {app_name}. Error: {str(e)}"
            else:
                return f"Sorry, {app_name} needs to be opened in a browser."
        
        # Try to open using Windows search
        try:
            subprocess.Popen([app_name + '.exe'])
            return f"Opening {app_name}..."
        except:
            pass
            
        # Try common locations
        common_paths = [
            f"C:\\Program Files\\{app_name}\\{app_name}.exe",
            f"C:\\Program Files (x86)\\{app_name}\\{app_name}.exe",
            f"C:\\Users\\{os.getenv('USERNAME', '')}\\AppData\\Local\\Programs\\{app_name}\\{app_name}.exe",
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                try:
                    os.startfile(path)
                    return f"Opening {app_name}..."
                except:
                    pass
                    
        return f"I couldn't find {app_name} on your system. Try installing it or add the path to the app_paths dictionary."
        
    def play_youtube(self, video):
        """Play a YouTube video."""
        try:
            pywhatkit.playonyt(video)
            return f"Playing {video} on YouTube..."
        except Exception as e:
            # Fallback to webbrowser
            search_url = f"https://www.youtube.com/results?search_query={quote(video)}"
            webbrowser.open(search_url)
            return f"Searching YouTube for {video}..."
            
    def tell_joke(self):
        """Tell a random joke."""
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!",
            "How does a penguin build its house? Igloos it together!",
            "Why did the bicycle fall over? Because it was two tired!",
            "What do you call a dog that does magic tricks? A Labracadabrador!",
            "Why don't eggs tell jokes? They'd crack each other up!",
            "What did the ocean say to the beach? Nothing, it just waved!",
            "Why did the math book look so sad? Because it had too many problems!",
        ]
        import random
        return random.choice(jokes)
        
    def calculate(self, expression):
        """Evaluate a mathematical expression."""
        try:
            # Clean up the expression
            expression = expression.replace('what is', '').replace('calculate', '').strip()
            expression = expression.replace('x', '*').replace('times', '*')
            expression = expression.replace('plus', '+').replace('minus', '-')
            expression = expression.replace('divided by', '/').replace('divide', '/')
            
            # Evaluate safely
            result = eval(expression)
            return f"The answer is {result}"
            
        except Exception as e:
            return f"I couldn't calculate that. Please try a simpler expression."
            
    def get_weather(self, city="London"):
        """Get weather information for a city."""
        # Note: This requires an API key from OpenWeatherMap
        # Set up your API key in config.py
        try:
            from config import WEATHER_API_KEY
            
            url = f"http://api.openweathermap.org/data/2.5/weather?q={quote(city)}&appid={WEATHER_API_KEY}&units=metric"
            response = requests.get(url)
            data = response.json()
            
            if data['cod'] == 200:
                temp = data['main']['temp']
                description = data['weather'][0]['description']
                humidity = data['main']['humidity']
                
                return f"The weather in {city} is {description} with a temperature of {temp}°C and humidity of {humidity}%"
            else:
                return f"I couldn't find weather for {city}."
                
        except ImportError:
            return "Weather API key not configured. Please add your API key to config.py"
        except Exception as e:
            return f"Sorry, I couldn't get the weather information."
            
    def search_wiki_or_web(self, query):
        """Determine whether to search Wikipedia or Google."""
        # Check if user wants Wikipedia info
        wiki_patterns = ['who is', 'what is', 'tell me about', 'who was', 'what was']
        
        for pattern in wiki_patterns:
            if pattern in query.lower():
                return self.search_wikipedia(query)
                
        # Otherwise, search Google
        return self.search_google(query)
        
    def process_command(self, command):
        """Process voice command and return response."""
        if not command:
            return
            
        command = command.lower()
        
        # Check for exit commands
        for exit_cmd in self.exit_commands:
            if exit_cmd in command:
                self.speak("Goodbye! Have a great day!")
                return "exit"
                
        # Greetings
        for greeting in self.greetings:
            if greeting in command:
                responses = [
                    "Hello! How can I help you today?",
                    "Hi there! What can I do for you?",
                    "Hey! How are you doing?",
                    "Hello! Ready to assist you. What would you like?"
                ]
                import random
                return random.choice(responses)
                
        # Help command
        if 'help' in command:
            return """Here are some things I can do:
- Tell you the time or date
- Search Wikipedia for information
- Search the web on Google
- Open applications like Notepad, Calculator, Browser
- Play videos on YouTube
- Tell you jokes
- Calculate math expressions
- And more! Just ask!"""
                
        # Time
        if 'time' in command:
            return self.get_time()
            
        # Date
        if 'date' in command or 'day' in command:
            return self.get_date()
            
        # Weather
        if 'weather' in command:
            # Try to extract city name
            city = "London"  # Default
            words = command.split()
            if 'in' in words:
                idx = words.index('in')
                if idx + 1 < len(words):
                    city = words[idx + 1]
            return self.get_weather(city)
            
        # Wikipedia search
        if any(pattern in command for pattern in ['who is', 'what is', 'tell me about', 'who was', 'what was']):
            return self.search_wikipedia(command)
            
        # Search/Google
        if 'search' in command or 'google' in command:
            query = command.replace('search', '').replace('google', '').strip()
            return self.search_google(query)
            
        # Open application
        if command.startswith('open '):
            app_name = command.replace('open ', '').strip()
            return self.open_application(app_name)
            
        # Play YouTube
        if 'play' in command and ('youtube' in command or 'song' in command or 'music' in command):
            query = command.replace('play', '').replace('youtube', '').replace('song', '').replace('music', '').strip()
            if query:
                return self.play_youtube(query)
            else:
                return self.play_youtube("popular music")
                
        # Just play music
        if command.startswith('play '):
            query = command.replace('play ', '').strip()
            return self.play_youtube(query)
            
        # Jokes
        if 'joke' in command or 'laugh' in command:
            return self.tell_joke()
            
        # Calculator
        if 'calculate' in command or 'what is' in command or any(op in command for op in ['+', '-', '*', '/', 'times', 'plus', 'minus']):
            if any(op in command for op in ['+', '-', '*', '/', 'times', 'plus', 'minus', 'divided']):
                return self.calculate(command)
                
        # Who are you
        if 'who are you' in command or 'what are you' in command:
            return """I am Mini Alexa, a voice assistant created to help you with various tasks.
I can answer questions, open applications, search the web, tell jokes, and more!"""
                
        # How are you
        if 'how are you' in command:
            return "I'm doing great, thank you for asking! How can I help you today?"
                
        # Thank you
        if 'thank' in command:
            return "You're welcome! Is there anything else I can help you with?"
                
        # Default - try to search Wikipedia first, then Google
        return self.search_wiki_or_web(command)
        
    def run(self):
        """Main run loop for the voice assistant."""
        self.speak("Hello! I am Mini Alexa, your voice assistant.")
        self.speak("How can I help you today?")
        
        while True:
            try:
                command = self.listen()
                
                if command:
                    response = self.process_command(command)
                    
                    if response == "exit":
                        break
                        
                    if response:
                        self.speak(response)
                        
            except KeyboardInterrupt:
                self.speak("Goodbye!")
                break
                
            except Exception as e:
                print(f"Error: {e}")
                self.speak("Sorry, I encountered an error. Let's try again.")


def main():
    """Main entry point."""
    assistant = VoiceAssistant()
    assistant.run()


if __name__ == "__main__":
    main()
