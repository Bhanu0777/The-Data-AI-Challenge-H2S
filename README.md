# The-Data-AI-Challenge-H2S

Redrob AI Challenge — Intelligent Candidate Discovery & Ranking

## Overview

This project ranks candidates for the **Senior AI Engineer — Founding Team** position at Redrob AI. It implements a multi-component scoring system that evaluates candidates based on their AI/ML skills, career trajectory, experience, availability, and education.

## Job Context

- **Role**: Senior AI Engineer – Founding Team
- **Company**: Redrob AI (Series A AI-native talent intelligence platform)
- **Location**: Pune/Noida, India (Hybrid)
- **Experience**: 5–9 years preferred
- **Focus**: Embeddings, retrieval, ranking, LLMs, fine-tuning, evaluation frameworks

## Scoring Architecture

Five weighted components with a behavioral modifier:

| Component | Weight | Description |
|-----------|--------|-------------|
| AI/ML Skill Match | 35% | Depth-first skill scoring (proficiency × duration × endorsements) |
| Title + Career Depth | 25% | Title relevance + product vs services company ratio |
| Experience Years | 15% | Years of experience with optimal range 6-8 |
| Availability/Engagement | 15% | Open to work, activity recency, notice period, location |
| Education | 10% | Tier-weighted education with field relevance multiplier |

**Behavioral Modifier**: Multiplies final score using recruiter response rate, interview completion, GitHub activity, and other engagement signals. Range: [0.30, 1.20]

## Files

- `rank.py` - Main candidate ranking implementation
- `validate_submission.py` - Submission validation script
- `candidates.jsonl` - Input candidate data (100K+ candidates)
- `submission.csv` - Output ranked candidates (top 100)
- `candidate_schema.json` - JSON schema for candidate data
- `requirements.txt` - Python dependencies (only python-docx for dev)

## Usage

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --top-n 100
```

## Key Design Decisions

- **Skill depth beats keyword count**: Proficiency score × duration × endorsement bonus
- **Platform assessment scores** (verified) weighted 2× self-reported
- **Consulting-only career** → hard cap at 0.25 component score
- **Honeypot detection** for profiles with impossible timelines or fabricated signals
- **Tie-break**: candidate_id ascending order

## Honeypot Detection

Profiles flagged as honeypots receive a score of 0.001. Detection criteria include:
- Years of experience inconsistent with career dates
- Expert proficiency in many skills with zero duration
- All skills expert/advanced but no endorsements
- Perfect (100) assessment scores on all skills

## Validation

```bash
python validate_submission.py
```

## Runtime

Approximately 45 seconds for 100K candidates on a modern CPU.