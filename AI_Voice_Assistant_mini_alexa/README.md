# Mini Alexa - AI Voice Assistant

A hands-free AI voice assistant that can understand speech, answer questions, open applications, and search Google.

## Project Structure

```
AI_Voice_Assistant_mini_alexa/
├── README.md              # Project documentation (this file)
├── requirements.txt       # Python dependencies
├── voice_assistant.py     # Main voice assistant application
├── app.py                 # Flask web interface
├── config.py              # Configuration settings
└── templates/
    └── index.html         # Web UI template
```

### File Descriptions

| File | Description |
|------|-------------|
| `README.md` | Contains project documentation, installation instructions, usage guide, and troubleshooting tips |
| `requirements.txt` | Lists all Python packages needed to run the project (SpeechRecognition, pyttsx3, wikipedia, etc.) |
| `voice_assistant.py` | Main application - contains all voice recognition, NLP processing, and command handling logic |
| `app.py` | Flask web server that provides a browser-based interface to interact with the voice assistant |
| `config.py` | Configuration file for API keys, application paths, and assistant settings |
| `templates/index.html` | HTML/CSS/JS frontend for the web interface |

## Features

- **Speech to Text**: Convert spoken words to text using speech recognition
- **Answer Questions**: Process natural language queries and provide intelligent responses
- **Open Applications**: Launch installed applications by voice command
- **Google Search**: Search the web for information using voice commands
- **Text to Speech**: Speak responses back to the user

## Requirements

- Python 3.8+
- Microphone
- Internet connection (for speech recognition and web search)

## Installation

1. Clone or download this repository
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For Windows, you may need to install pyttsx3 dependencies:
   - SAPI5 (comes with Windows)
   
4. For Linux, you may need to install:
   ```bash
   sudo apt-get install espeak ffmpeg libespeak1
   ```

## Usage

Run the voice assistant:
```bash
python voice_assistant.py
```

### Available Commands

- **Greetings**: "hello", "hi", "hey", "good morning", "good evening"
- **Time**: "what time is it", "tell me the time"
- **Date**: "what date is it", "what day is it today"
- **Weather**: "what's the weather" (requires API key)
- **Search**: "search for [query]" or "google [query]"
- **Open Apps**: "open [application name]" (e.g., "open notepad", "open browser", "open youtube")
- **Play Music**: "play music" or "play [song name]"
- **Jokes**: "tell me a joke", "make me laugh"
- **Wikipedia**: "who is [person]", "what is [thing]"
- **Math**: "calculate [expression]" or "what is [math]"
- **Exit**: "exit", "quit", "goodbye", "sleep"

### Example Interactions

```
You: "Hello"
Assistant: "Hello! How can I help you today?"

You: "What's the time?"
Assistant: "The current time is 2:30 PM"

You: "Search for Python tutorials"
Assistant: "Searching for Python tutorials..."

You: "Open notepad"
Assistant: "Opening Notepad..."

You: "Who is Elon Musk?"
Assistant: "Elon Musk is a entrepreneur and businessman..."
```

## Configuration

### Weather API (Optional)

To enable weather features, sign up for a free API key at [OpenWeatherMap](https://openweathermap.org/api) and add it to `config.py`:

```python
WEATHER_API_KEY = "your_api_key_here"
```

### Custom Commands

You can add custom commands by editing the `COMMANDS` dictionary in `voice_assistant.py`:

```python
COMMANDS = {
    "your_command": your_handler_function,
    ...
}
```

## Troubleshooting

### Speech Recognition Issues
- Make sure your microphone is properly connected and configured
- Check if your OS has the correct microphone permissions
- For better accuracy, ensure you're in a quiet environment

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- On Windows, verify pyttsx3 is working properly

### Application Launching Issues
- Ensure the application paths in the code match your system
- Some applications may require administrator privileges

## License

MIT License

## Author

Created as a mini Alexa voice assistant demonstration project.
