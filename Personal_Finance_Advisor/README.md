# Personal Finance Advisor AI 🤖💰

An AI-powered personal finance advisor that analyzes your spending patterns and provides personalized savings recommendations.

## Features

- **Spending Pattern Analysis**: Automatically categorizes and analyzes your expenses
- **AI Recommendations**: Get intelligent suggestions based on the 50/30/20 budget rule
- **Savings Plan Generator**: Personalized step-by-step plan to reach your savings goals
- **Category Breakdown**: Visual breakdown of spending across different categories
- **Potential Savings Calculator**: See how much you could save by optimizing spending

## Categories Analyzed

- 🏠 Housing (30% ideal)
- 🍔 Food (15% ideal)
- 🚗 Transportation (10% ideal)
- 💡 Utilities (10% ideal)
- 🎬 Entertainment (10% ideal)
- 🛒 Shopping (10% ideal)
- 🏥 Healthcare (5% ideal)
- 💰 Savings (20% ideal)
- 📦 Other (5% ideal)

## Installation

1. Navigate to the project directory:
```bash
cd Personal_Finance_Advisor
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Enter your monthly income and expenses
4. Click "Analyze My Finances" to get your personalized recommendations

## How It Works

### Spending Analysis
The AI analyzes your spending by:
- Calculating your savings rate (percentage of income saved)
- Comparing each category against ideal budget percentages
- Identifying areas where you're overspending
- Highlighting categories that need attention

### Recommendations Engine
The system provides recommendations based on:
- **Savings Rate**: Urgent/warning/success messages based on your savings percentage
- **Category Analysis**: Specific tips for high-spending categories
- **Automation**: Suggestions for automating savings
- **Emergency Fund**: Goals for building financial security

### Savings Plan Generator
Creates a personalized action plan:
1. Calculates target savings (default: 20% of income)
2. Identifies reducible expenses
3. Generates step-by-step actionable items
4. Shows monthly and yearly potential savings

## Example Output

After entering your financial data, you'll see:
- 📊 Financial Summary (income, savings, expenses)
- 📈 Visual category breakdown with status indicators
- 💡 AI-powered recommendations
- 🎯 Personalized savings action plan
- 💰 Potential savings opportunities

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Design**: Modern responsive UI with CSS animations
- **Logic**: Rule-based AI system using the 50/30/20 budget principle

## Budget Philosophy

This app uses the **50/30/20 Rule**:
- **50%** Needs (Housing, Food, Transportation, Utilities, Healthcare)
- **30%** Wants (Entertainment, Shopping, Other)
- **20%** Savings (Emergency fund, Investments, Debt repayment)

## License

MIT License

---

Built with ❤️ for better financial health
