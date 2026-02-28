"""
Configuration file for Mini Alexa Voice Assistant
"""

# OpenWeatherMap API Key (optional - for weather functionality)
# Get your free API key at: https://openweathermap.org/api
# Sign up for an account, go to API keys, and copy your key here
WEATHER_API_KEY = ""  # Add your API key here

# Assistant settings
ASSISTANT_NAME = "Mini Alexa"
VOICE_RATE = 150  # Speed of speech (words per minute)
VOICE_VOLUME = 1.0  # Volume level (0.0 to 1.0)

# Application paths (customize for your system)
# Add more applications as needed
APP_PATHS = {
    'notepad': 'notepad.exe',
    'calculator': 'calc.exe',
    'browser': '',
    'chrome': '',
    'firefox': '',
    'microsoft edge': '',
    'word': '',
    'excel': '',
    'powerpoint': '',
    'spotify': '',
    'vscode': 'C:\\Users\\' + __import__('os').getlogin() + '\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe',
    'pycharm': 'C:\\Program Files\\JetBrains\\PyCharm Community Edition\\bin\\pycharm64.exe',
}

# Web applications (open in browser)
WEB_APPS = {
    'youtube': 'https://www.youtube.com',
    'whatsapp': 'https://web.whatsapp.com',
    'gmail': 'https://www.gmail.com',
    'facebook': 'https://www.facebook.com',
    'twitter': 'https://www.twitter.com',
    'instagram': 'https://www.instagram.com',
    'reddit': 'https://www.reddit.com',
    'google': 'https://www.google.com',
    'netflix': 'https://www.netflix.com',
    'linkedin': 'https://www.linkedin.com',
    'github': 'https://www.github.com',
}

# Speech recognition language
LANGUAGE = 'en-US'
