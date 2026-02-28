#!/usr/bin/env python
"""Test script to verify resume matching works correctly."""

from resume_screening import ResumeScreener

# Test with identical content
resume = 'Software Engineer Python Java JavaScript React Node SQL MongoDB AWS'
job = 'Software Engineer Python Java JavaScript React Node SQL MongoDB AWS'

model = ResumeScreener()
model.fit([resume], [job])
score = model.calculate_match_score(resume, job)
print('Identical content match:', score, '%')

# Test with similar content
resume2 = 'Software Developer Python JavaScript React Node SQL AWS'
job2 = 'Software Engineer Python Java React Node SQL AWS'

score2 = model.calculate_match_score(resume2, job2)
print('Similar content match:', score2, '%')

# Test with completely different content
resume3 = 'Chef cooking food restaurant recipes'
job3 = 'Software Engineer Python Java programming'

score3 = model.calculate_match_score(resume3, job3)
print('Different content match:', score3, '%')

# Test with detailed analysis
print("\n--- Detailed Analysis Test ---")
analysis = model.get_detailed_analysis(resume2, job2)
print('Match Score:', analysis['match_score'], '%')
print('Keyword Match:', analysis['keyword_match_percentage'], '%')
print('Skill Match:', analysis['skill_match_percentage'], '%')
print('Matching Skills:', analysis['matching_skills'])
print('Missing Skills:', analysis['missing_skills'])
