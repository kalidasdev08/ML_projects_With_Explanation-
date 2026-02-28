# AI Interview Bot 🤖

An AI-powered interview practice application that generates unique interview questions using Google Gemini API and evaluates your answers.

## Features

- **AI-Generated Questions**: Unique questions created dynamically using Gemini AI
- **Job-Role Customization**: Enter your target job role for personalized questions
- **Multiple Categories**: Technical, Behavioral, Situational, and General questions
- **AI-Powered Evaluation**: Detailed feedback on clarity, relevance, depth, and specificity
- **Interactive UI**: Clean and modern web interface for practicing interviews

## Prerequisites

- Python 3.8+
- Google Gemini API Key (required for full functionality)

## Installation

1. Navigate to the project directory:
```bash
cd AI_Interview_Bot
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Gemini API key:
```bash
# On macOS/Linux
export GEMINI_API_KEY="your_api_key_here"

# On Windows (Command Prompt)
set GEMINI_API_KEY=your_api_key_here

# On Windows (PowerShell)
$env:GEMINI_API_KEY="your_api_key_here"
```

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key and set it as an environment variable

## Running the Application

```bash
python app.py
```

The application will start on `http://localhost:5000`. Open this URL in your web browser.

## How to Use

1. **Enter Job Role** (Optional): Type your target job role (e.g., "Software Engineer", "Data Scientist") for personalized questions
   - OR select a specific category (Technical, Behavioral, Situational, General)
   
2. **Start Interview**: Click "Start Interview" to begin

3. **Answer Questions**: Type your answer to each unique AI-generated question

4. **Get Feedback**: Submit your answer to receive detailed AI evaluation

5. **Review Results**: Complete all 5 questions to see your overall performance

## Project Structure

```
AI_Interview_Bot/
├── app.py                 # Flask web application
├── answer_evaluator.py    # AI question generation & evaluation
├── questions.json         # Fallback questions database
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend UI
└── README.md             # This file
```

## How It Works

1. **Question Generation**: Uses Gemini AI to create unique, original interview questions
   - If a job role is provided, generates role-specific questions
   - Otherwise, creates questions from selected category or mix of all categories

2. **Answer Evaluation**: Gemini AI evaluates answers based on:
   - **Clarity (25%)**: How clearly is the answer communicated?
   - **Relevance (30%)**: How well does it address the question?
   - **Depth (25%)**: How detailed and thorough is the response?
   - **Specificity (20%)**: Does it include specific examples?

3. **Feedback**: Provides actionable feedback with strengths and areas for improvement

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Your Google Gemini API key (required for AI features) |
| `PORT` | Server port (default: 5000) |
| `FLASK_DEBUG` | Set to 'true' for debug mode |

## Without API Key

If no API key is provided, the application will use basic fallback questions and simple text analysis for evaluation. For the full AI experience, always provide a valid Gemini API key.

## Technologies Used

- **Backend**: Flask (Python)
- **AI**: Google Gemini API
- **Frontend**: HTML, CSS, JavaScript

## License

MIT License
