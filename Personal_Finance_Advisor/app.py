from flask import Flask, render_template, request
import os

app = Flask(__name__)

# Spending categories with typical budget percentages (50/30/20 rule baseline)
CATEGORIES = {
    'Housing': {'icon': '🏠', 'ideal_pct': 30, 'color': '#FF6B6B'},
    'Food': {'icon': '🍔', 'ideal_pct': 15, 'color': '#4ECDC4'},
    'Transportation': {'icon': '🚗', 'ideal_pct': 10, 'color': '#45B7D1'},
    'Utilities': {'icon': '💡', 'ideal_pct': 10, 'color': '#96CEB4'},
    'Entertainment': {'icon': '🎬', 'ideal_pct': 10, 'color': '#FFEAA7'},
    'Shopping': {'icon': '🛒', 'ideal_pct': 10, 'color': '#DDA0DD'},
    'Healthcare': {'icon': '🏥', 'ideal_pct': 5, 'color': '#98D8C8'},
    'Savings': {'icon': '💰', 'ideal_pct': 20, 'color': '#70A1FF'},
    'Other': {'icon': '📦', 'ideal_pct': 5, 'color': '#A8A8A8'}
}

def analyze_spending_patterns(income, expenses):
    """Analyze spending patterns and generate savings recommendations"""
    
    total_expenses = sum(expenses.values())
    savings = income - total_expenses
    savings_rate = (savings / income * 100) if income > 0 else 0
    
    # Category breakdown
    category_analysis = []
    for cat, data in CATEGORIES.items():
        amount = expenses.get(cat, 0)
        pct = (amount / income * 100) if income > 0 else 0
        ideal_amount = income * (data['ideal_pct'] / 100)
        diff = amount - ideal_amount
        
        status = 'good' if abs(diff) <= ideal_amount * 0.1 else ('high' if diff > 0 else 'low')
        
        category_analysis.append({
            'name': cat,
            'icon': data['icon'],
            'color': data['color'],
            'amount': amount,
            'percentage': pct,
            'percentage_capped': min(pct, 100),
            'ideal_pct': data['ideal_pct'],
            'ideal_amount': ideal_amount,
            'difference': diff,
            'status': status
        })
    
    # Sort by percentage spent
    category_analysis.sort(key=lambda x: x['percentage'], reverse=True)
    
    # Generate AI recommendations
    recommendations = generate_recommendations(income, expenses, savings_rate, category_analysis)
    
    # Calculate potential savings
    potential_savings = calculate_potential_savings(expenses, category_analysis)
    
    return {
        'income': income,
        'total_expenses': total_expenses,
        'savings': savings,
        'savings_rate': savings_rate,
        'categories': category_analysis,
        'recommendations': recommendations,
        'potential_savings': potential_savings
    }

def generate_recommendations(income, expenses, savings_rate, category_analysis):
    """Generate personalized savings recommendations"""
    recommendations = []
    
    # Check savings rate
    if savings_rate < 10:
        recommendations.append({
            'type': 'urgent',
            'icon': '⚠️',
            'title': 'Low Savings Rate',
            'message': f'Your savings rate is only {savings_rate:.1f}%. Aim for at least 20% to build financial security.'
        })
    elif savings_rate < 20:
        recommendations.append({
            'type': 'warning',
            'icon': '📈',
            'title': 'Moderate Savings',
            'message': f'Good start! {savings_rate:.1f}% savings rate is decent. Try to increase to 20% for better financial health.'
        })
    else:
        recommendations.append({
            'type': 'success',
            'icon': '🎉',
            'title': 'Excellent Savings',
            'message': f'Amazing! {savings_rate:.1f}% savings rate puts you in great financial shape!'
        })
    
    # Check for high spending categories
    for cat in category_analysis:
        if cat['status'] == 'high' and cat['difference'] > income * 0.05:
            recommendations.append({
                'type': 'tip',
                'icon': cat['icon'],
                'title': f'High {cat["name"]} Spending',
                'message': f'Your {cat["name"]} expenses are {cat["percentage"]:.1f}% of income (ideal: {cat["ideal_pct"]}%). '
                          f'Consider reducing by ${cat["difference"]:.0f}/month to reach the recommended level.'
            })
    
    # Check for missing savings
    if expenses.get('Savings', 0) < income * 0.20:
        recommendations.append({
            'type': 'tip',
            'icon': '💡',
            'title': 'Automate Your Savings',
            'message': 'Set up automatic transfers to savings right after payday. "Pay yourself first" is the key to building wealth!'
        })
    
    # Check for emergency fund
    total_expenses = sum(expenses.values())
    monthly_expenses = total_expenses
    recommendations.append({
        'type': 'info',
        'icon': '🏦',
        'title': 'Emergency Fund Goal',
        'message': f'Aim to build an emergency fund equal to 3-6 months of expenses (${monthly_expenses * 3:.0f}-${monthly_expenses * 6:.0f})'
    })
    
    # Specific category tips
    if expenses.get('Entertainment', 0) > income * 0.10:
        recommendations.append({
            'type': 'tip',
            'icon': '🎮',
            'title': 'Entertainment Budget',
            'message': 'Consider setting a monthly entertainment budget. Try the "1-day no-spend" challenge each week!'
        })
    
    if expenses.get('Shopping', 0) > income * 0.10:
        recommendations.append({
            'type': 'tip',
            'icon': '🛍️',
            'title': 'Smart Shopping Tips',
            'message': 'Try the 24-hour rule: wait 24 hours before any non-essential purchase to avoid impulse buying.'
        })
    
    if expenses.get('Food', 0) > income * 0.20:
        recommendations.append({
            'type': 'tip',
            'icon': '🥗',
            'title': 'Food Expense Optimization',
            'message': 'Meal planning and cooking at home can save you significantly. Try batch cooking on weekends!'
        })
    
    return recommendations

def calculate_potential_savings(expenses, category_analysis):
    """Calculate potential savings if spending is optimized"""
    total_potential = 0
    savings_tips = []
    
    for cat in category_analysis:
        if cat['status'] == 'high' and cat['difference'] > 0:
            potential = cat['difference'] * 0.5  # Suggest 50% reduction
            total_potential += potential
            savings_tips.append({
                'category': cat['name'],
                'icon': cat['icon'],
                'potential': potential,
                'tip': f'Reduce {cat["name"]} spending by {cat["difference"]:.0f}'
            })
    
    return {
        'total': total_potential,
        'tips': savings_tips
    }

def generate_savings_plan(income, expenses, target_savings_rate=20):
    """Generate a personalized savings plan"""
    current_savings = income - sum(expenses.values())
    target_savings = income * (target_savings_rate / 100)
    shortfall = target_savings - current_savings
    
    if shortfall <= 0:
        return {
            'status': 'exceeds',
            'message': 'You already meet your savings goal! Consider increasing your savings rate or investing.',
            'steps': []
        }
    
    # Generate step-by-step plan
    steps = []
    
    # Find categories where spending can be reduced
    reducible = []
    for cat_name, amount in expenses.items():
        if cat_name != 'Savings' and amount > 0:
            ideal_pct = CATEGORIES.get(cat_name, {}).get('ideal_pct', 10)
            ideal_amount = income * (ideal_pct / 100)
            if amount > ideal_amount:
                reducible.append({
                    'category': cat_name,
                    'current': amount,
                    'ideal': ideal_amount,
                    'savings': amount - ideal_amount
                })
    
    reducible.sort(key=lambda x: x['savings'], reverse=True)
    
    total_reduction = 0
    for item in reducible:
        if total_reduction >= shortfall:
            break
        steps.append({
            'action': f'Reduce {item["category"]} spending',
            'amount': item['savings'],
            'from': item['current'],
            'to': item['ideal']
        })
        total_reduction += item['savings']
    
    if total_reduction < shortfall:
        remaining = shortfall - total_reduction
        steps.append({
            'action': 'Find additional income source',
            'amount': remaining,
            'from': 0,
            'to': remaining
        })
    
    return {
        'status': 'plan',
        'target_savings': target_savings,
        'current_savings': current_savings,
        'shortfall': shortfall,
        'monthly_steps': steps,
        'yearly_potential': target_savings * 12
    }

@app.route('/')
def index():
    return render_template('index.html', categories=CATEGORIES)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Get income
        income = float(request.form.get('income', 0))
        
        # Get expenses for each category
        expenses = {}
        for cat in CATEGORIES.keys():
            expenses[cat] = float(request.form.get(f'expense_{cat}', 0))
        
        # Analyze spending patterns
        analysis = analyze_spending_patterns(income, expenses)
        
        # Generate savings plan
        savings_plan = generate_savings_plan(income, expenses)
        
        return render_template('index.html', 
                             categories=CATEGORIES,
                             analysis=analysis,
                             savings_plan=savings_plan,
                             expenses=expenses,
                             income=income)
    except Exception as e:
        return render_template('index.html', 
                             categories=CATEGORIES,
                             error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
