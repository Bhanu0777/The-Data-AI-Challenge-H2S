#!/usr/bin/env python3
"""
Redrob AI Challenge — Intelligent Candidate Discovery & Ranking
Job: Senior AI Engineer – Founding Team @ Redrob AI

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Scoring Architecture
====================
Five weighted components × behavioural modifier:

  Component                 Weight
  ─────────────────────────────────
  1. AI/ML skill match        35%
  2. Title + career depth     25%
  3. Experience years         15%
  4. Availability/engagement  15%
  5. Education                10%

  final = clamp(component_score × behavioural_modifier)

Key design decisions:
  - Skill depth beats keyword count: proficiency × duration × endorsements
  - Platform assessment scores (verified) weighted 2× self-reported
  - Consulting-only career → hard cap at 0.25 component score
  - Honeypot detection: impossible timelines, high-skill + zero-duration combos
  - Behavioural modifier maps [0,1] → [0.3, 1.2] to avoid zeroing good profiles
  - Tie-break: candidate_id ascending (per spec)

Runtime: ~45s for 100K candidates on a modern CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Reference date
# ─────────────────────────────────────────────────────────────────────────────

TODAY = date(2026, 6, 8)  # fixed for reproducibility

# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary tables
# ─────────────────────────────────────────────────────────────────────────────

# Every skill name normalised to lowercase.
# Two tiers: PREMIUM = absolutely required by the JD; CORE = strongly desired.

AI_PREMIUM = frozenset({
    # Embeddings & retrieval (must-have cluster)
    "embeddings", "sentence transformers", "sentence-transformers",
    "text embeddings", "vector search", "vector database", "vector db",
    "semantic search", "dense retrieval", "hybrid search", "bi-encoder",
    "cross-encoder", "bge", "e5", "openai embeddings",
    # Vector stores
    "faiss", "milvus", "weaviate", "qdrant", "pinecone",
    "opensearch", "elasticsearch", "annoy", "scann",
    # IR & ranking
    "bm25", "information retrieval", "learning to rank", "ltr",
    "ndcg", "mrr", "map", "precision@k", "recall@k",
    "recommendation systems", "recommender systems", "collaborative filtering",
    "ranking systems", "retrieval augmented generation", "rag",
    # LLM core
    "llm", "llms", "large language model", "large language models",
    "fine-tuning llms", "fine-tuning", "finetuning", "instruction tuning",
    "rlhf", "lora", "qlora", "peft", "adapter tuning",
    "prompt engineering", "chain of thought",
    # Core NLP
    "nlp", "natural language processing", "text classification",
    "named entity recognition", "ner", "question answering", "summarisation",
    "summarization", "text generation",
    # Evaluation
    "a/b testing", "offline evaluation", "online evaluation",
    "evaluation framework", "model evaluation",
})

AI_CORE = frozenset({
    # ML frameworks
    "pytorch", "tensorflow", "keras", "jax",
    "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
    "huggingface", "hugging face", "transformers",
    # MLOps
    "mlflow", "weights & biases", "wandb", "bentoml", "ray", "ray serve",
    "kubeflow", "dvc", "model serving", "feature store", "triton",
    "model deployment", "model monitoring", "feature engineering",
    # Data
    "spark", "pyspark", "airflow", "kafka", "databricks",
    "data pipelines", "etl", "data engineering",
    # Computer science core
    "python", "distributed systems", "rest api", "microservices",
    "docker", "kubernetes",
    # Adjacent AI
    "statistical modeling", "time series", "anomaly detection",
    "gradient boosting", "neural networks", "deep learning",
    "machine learning", "data science",
    "image classification", "object detection", "computer vision",
    "speech recognition", "tts", "gans", "generative ai",
    "reinforcement learning",
    "apache beam", "milvus", "apache flink",
})

# Consulting/services firms — full career there is a soft disqualifier per JD
BIG_SERVICES = frozenset({
    "tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "hcl technologies", "tech mahindra", "hexaware", "mphasis",
    "l&t infotech", "l&t technology services", "mindtree",
    "niit technologies", "zensar", "cyient", "coforge", "mastech",
    "kforce", "globant", "epam",
})

# Preferred locations per JD (Noida/Pune primary; Hyd/Mum/Delhi welcome)
PREFERRED_LOCATIONS = frozenset({
    "noida", "pune", "hyderabad", "mumbai", "delhi", "gurugram", "gurgaon",
    "bengaluru", "bangalore", "new delhi", "delhi ncr", "ncr",
    "greater noida", "faridabad",
})

# Title categories
_AI_ML_TITLES = frozenset({
    "ai engineer", "ml engineer", "machine learning engineer",
    "nlp engineer", "applied scientist", "research engineer",
    "data scientist", "senior ai engineer", "staff ml engineer",
    "principal engineer", "founding engineer", "ai researcher",
    "deep learning engineer", "recommendation engineer",
    "search engineer", "information retrieval engineer",
    "backend engineer", "senior backend engineer",
    "software engineer", "senior software engineer",
    "data engineer", "analytics engineer",
    "platform engineer", "infrastructure engineer",
    "full stack engineer", "fullstack engineer",
    "junior ml engineer", "senior machine learning engineer",
    "senior data scientist",
})

_NEUTRAL_TITLES = frozenset({
    "project manager", "product manager", "business analyst",
    "operations manager", "technical lead", "tech lead",
    "full stack", "fullstack",
})

_WEAK_TITLES = frozenset({
    "marketing manager", "hr manager", "human resource", "sales executive",
    "accountant", "customer support", "content writer", "graphic designer",
    "civil engineer", "mechanical engineer", "finance manager",
    "supply chain", "procurement",
})

# Proficiency → numeric
PROF_MAP = {
    "beginner": 0.25,
    "intermediate": 0.55,
    "advanced": 0.80,
    "expert": 1.00,
}

# Education tier → numeric
EDU_TIER = {
    "tier_1": 1.00,  # IITs, IISc, BITS Pilani, top IIMs
    "tier_2": 0.75,  # NITs, IIIT-H, good state universities
    "tier_3": 0.50,  # Mid-tier engineering colleges
    "tier_4": 0.30,  # Lower-tier colleges
    "unknown": 0.40,
}

# Fields relevant to AI/ML/CS roles
RELEVANT_FIELDS = frozenset({
    "computer science", "cs", "information technology", "it",
    "artificial intelligence", "machine learning", "data science",
    "statistics", "mathematics", "math", "electrical engineering",
    "electronics", "software engineering", "computational linguistics",
    "operations research",
})


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.lower().strip() if s else ""


def _months_ago(date_str: str | None) -> float:
    """Months between date_str and TODAY. Returns 999 for missing/invalid."""
    if not date_str:
        return 999.0
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return max(0.0, (TODAY - d).days / 30.44)
    except (ValueError, TypeError):
        return 999.0


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else (hi if v > hi else v)


def _logistic(x: float, midpoint: float = 5.0, k: float = 0.8) -> float:
    """Sigmoid: maps any real to (0, 1), useful for smooth penalties."""
    return 1.0 / (1.0 + math.exp(-k * (x - midpoint)))


def _is_services(company: str) -> bool:
    n = _norm(company)
    return any(f in n for f in BIG_SERVICES)


def _skill_tiers(name: str) -> tuple[bool, bool]:
    """Return (is_core, is_premium)."""
    n = _norm(name)
    is_premium = any(kw in n or n in kw for kw in AI_PREMIUM)
    is_core = is_premium or any(kw in n or n in kw for kw in AI_CORE)
    return is_core, is_premium


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot detection
# ─────────────────────────────────────────────────────────────────────────────

def _is_honeypot(candidate: dict) -> bool:
    """
    Return True if the profile shows impossible/suspicious signals.
    Spec §7: ~80 honeypots with subtly impossible profiles.
    These are forced to relevance tier 0 in the ground truth.
    """
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    yoe = profile.get("years_of_experience", 0)

    # Check 1: career timeline impossible — start_date before reasonable
    for role in career:
        start = role.get("start_date", "")
        if start:
            try:
                start_yr = int(start[:4])
                # If start_year means they'd have worked before age ~15
                # E.g. born ~2000, started work before 2015 at a company
                # founded recently
                pass
            except ValueError:
                pass

    # Check 2: years_of_experience inconsistent with career dates
    if career:
        earliest_start_months = 999.0
        for role in career:
            m = _months_ago(role.get("start_date"))
            if m < earliest_start_months:
                earliest_start_months = m
        # Career span in years
        career_span_years = earliest_start_months / 12.0
        # If stated YoE > career span + 2 years (big discrepancy)
        if yoe > career_span_years + 3 and career_span_years > 0:
            return True

    # Check 3: Expert in many skills with 0 months duration
    expert_zero_dur = sum(
        1 for s in skills
        if s.get("proficiency") in ("expert", "advanced")
        and s.get("duration_months", 0) == 0
    )
    if expert_zero_dur >= 4:
        return True

    # Check 4: All skills expert/advanced but no endorsements at all
    if len(skills) >= 6:
        all_high = all(s.get("proficiency") in ("expert", "advanced") for s in skills)
        total_endorsements = sum(s.get("endorsements", 0) for s in skills)
        if all_high and total_endorsements == 0:
            return True

    # Check 5: Assessment score of 100 on every skill (too perfect)
    assessments = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    if len(assessments) >= 3 and all(v == 100.0 for v in assessments.values()):
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Component 1 — AI/ML skill match  (weight 0.35)
# ─────────────────────────────────────────────────────────────────────────────

def _score_skills(candidate: dict) -> float:
    """
    Depth-first skill scoring.  Raw count is a trap (keyword stuffing).
    Formula per relevant skill:
        value = prof_score × duration_factor × (1 + endorsement_bonus)
        if assessed: replace prof_score with blend(0.65*assessed + 0.35*self)
    Premium skills get 1.6× weight vs core skills.
    Stuffing penalty: many beginner skills with no assessment + short duration.
    """
    skills: list[dict] = candidate.get("skills", [])
    assessed: dict = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {}) or {}
    )

    # Build normalised assessment lookup
    assessed_norm = {_norm(k): v / 100.0 for k, v in assessed.items()}

    premium_total = core_total = 0.0
    premium_max = core_max = 0.0
    stuffing_penalty = 0.0
    premium_count = 0

    for sk in skills:
        name = sk.get("name", "")
        prof = sk.get("proficiency", "beginner")
        endorse = sk.get("endorsements", 0)
        dur = sk.get("duration_months", 0)

        is_core, is_premium = _skill_tiers(name)
        if not is_core:
            continue

        if is_premium:
            premium_count += 1

        # Self-reported proficiency
        prof_score = PROF_MAP.get(prof, 0.25)

        # Platform-verified assessment (more trustworthy)
        sk_norm = _norm(name)
        assessed_val = None
        for ak, av in assessed_norm.items():
            if ak == sk_norm or sk_norm in ak or ak in sk_norm:
                assessed_val = av
                break

        if assessed_val is not None:
            effective = 0.65 * assessed_val + 0.35 * prof_score
        else:
            effective = prof_score

        # Duration factor: 0.3 (0 months) → 1.0 (18+ months) → 1.15 (36+ months)
        if dur == 0:
            dur_factor = 0.30
        elif dur < 6:
            dur_factor = 0.50
        elif dur < 12:
            dur_factor = 0.70
        elif dur < 24:
            dur_factor = 0.90
        elif dur < 36:
            dur_factor = 1.00
        else:
            dur_factor = 1.10

        # Endorsement bonus: log-scaled, max +0.25
        endorse_bonus = min(0.25, math.log1p(endorse) / 18.0)

        value = effective * dur_factor * (1.0 + endorse_bonus)

        # Keyword stuffing: beginner + no assessment + very short
        if prof == "beginner" and assessed_val is None and dur < 6:
            stuffing_penalty += 0.04

        if is_premium:
            premium_total += value
            premium_max += 1.10 * 1.6 * 1.25  # max value per premium skill
        else:
            core_total += value
            core_max += 1.10 * 1.25           # max value per core skill

    # Normalise each bucket separately
    p_score = premium_total / premium_max if premium_max > 0 else 0.0
    c_score = core_total / core_max if core_max > 0 else 0.0

    # Premium matters 70%, core 30%
    raw = 0.70 * p_score + 0.30 * c_score

    # Bonus for breadth of premium skills (up to +0.15)
    breadth_bonus = min(0.15, premium_count / 12.0)

    # Apply stuffing penalty (capped at -0.25)
    stuffing = min(stuffing_penalty, 0.25)

    return _clamp(raw + breadth_bonus - stuffing)


# ─────────────────────────────────────────────────────────────────────────────
# Component 2 — Title + career trajectory  (weight 0.25)
# ─────────────────────────────────────────────────────────────────────────────

def _score_career(candidate: dict) -> float:
    """
    Evaluates:
    • Current title relevance
    • Fraction of career at product vs services companies
    • ML/AI depth in role descriptions (recency-weighted)
    • Career progression (upward titles)
    Hard cap at 0.25 for full-consulting-career candidates (per JD).
    """
    profile = candidate.get("profile", {})
    career: list[dict] = candidate.get("career_history", [])
    title = _norm(profile.get("current_title", ""))

    # Title score
    if any(kw in title for kw in _AI_ML_TITLES):
        title_score = 0.95
    elif title in {"data analyst", "analytics engineer"}:
        title_score = 0.60
    elif any(kw in title for kw in _NEUTRAL_TITLES):
        title_score = 0.30
    elif any(kw in title for kw in _WEAK_TITLES):
        title_score = 0.05
    else:
        title_score = 0.35

    if not career:
        return _clamp(title_score * 0.5)

    # Services vs product
    services_count = sum(1 for r in career if _is_services(r.get("company", "")))
    total = len(career)
    full_consulting = services_count == total

    product_ratio = (total - services_count) / total if total > 0 else 0.0
    product_mult = 0.45 + 0.55 * product_ratio  # 0.45 → 1.00

    # ML/AI in career descriptions — recency-weighted
    ml_keywords = {
        "machine learning", "ml ", " ml,", "nlp", "neural", "embedding",
        "retrieval", "ranking", "recommendation", "deep learning",
        "llm", "fine-tun", "transformer", "vector", "semantic",
        "pytorch", "tensorflow", "sklearn", "xgboost", "model",
        "rag", "retrieval augmented", "information retrieval",
        "a/b test", "evaluation", "ndcg", "mrr",
    }

    weighted_ml = 0.0
    weight_total = 0.0
    for role in career:
        desc = _norm(role.get("description", ""))
        role_title = _norm(role.get("title", ""))
        months_ago = _months_ago(role.get("start_date"))

        # Recency weight: roles starting within 3 years get full weight
        recency_w = max(0.2, 1.0 - months_ago / 72.0)

        ml_in_role = any(kw in desc or kw in role_title for kw in ml_keywords)
        weighted_ml += recency_w * (1.0 if ml_in_role else 0.0)
        weight_total += recency_w

    career_ml_score = weighted_ml / weight_total if weight_total > 0 else 0.0

    # Combine
    raw = title_score * 0.45 + career_ml_score * 0.40 + product_ratio * 0.15
    raw *= product_mult

    if full_consulting:
        raw = min(raw, 0.25)

    return _clamp(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Component 3 — Experience years  (weight 0.15)
# ─────────────────────────────────────────────────────────────────────────────

def _score_experience(candidate: dict) -> float:
    """
    JD wants 5-9 years (ideal 6-8).
    Disqualifiers: <2 years (too junior), >18 years (potential mismatch).
    Also checks for employment gap (no current role).
    """
    yoe = candidate.get("profile", {}).get("years_of_experience", 0)
    career: list[dict] = candidate.get("career_history", [])

    if yoe < 1:
        base = 0.05
    elif yoe < 3:
        base = 0.20 + (yoe - 1) * 0.08
    elif yoe < 5:
        base = 0.36 + (yoe - 3) * 0.12
    elif yoe <= 9:
        # Sweet spot; peak at 6-8
        if 6.0 <= yoe <= 8.0:
            base = 1.00
        elif 5.0 <= yoe < 6.0:
            base = 0.85 + (yoe - 5.0) * 0.15
        else:  # 8-9
            base = 0.90
    elif yoe <= 12:
        base = 0.75
    elif yoe <= 15:
        base = 0.60
    else:
        base = 0.45

    # Employment gap penalty
    has_current = any(r.get("is_current", False) for r in career)
    if not has_current and yoe > 3:
        # Check last role end date
        ends = [r.get("end_date") for r in career if r.get("end_date")]
        if ends:
            most_recent_end = min(_months_ago(e) for e in ends)
            if most_recent_end > 18:
                base *= 0.70  # Significant gap

    return _clamp(base)


# ─────────────────────────────────────────────────────────────────────────────
# Component 4 — Availability & engagement  (weight 0.15)
# ─────────────────────────────────────────────────────────────────────────────

def _score_availability(candidate: dict) -> float:
    """
    Combines:
    • open_to_work_flag
    • last_active_date recency (key: inactive >6 mo → near-zero)
    • notice_period_days (JD: ≤30 preferred)
    • location match
    • applications_submitted_30d (shows active job seeking)
    """
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    # Open to work
    open_score = 1.0 if sig.get("open_to_work_flag", False) else 0.40

    # Activity recency (most important availability signal)
    inactive_mo = _months_ago(sig.get("last_active_date"))
    if inactive_mo <= 1:
        active_score = 1.00
    elif inactive_mo <= 2:
        active_score = 0.90
    elif inactive_mo <= 3:
        active_score = 0.80
    elif inactive_mo <= 6:
        active_score = 0.60
    elif inactive_mo <= 9:
        active_score = 0.35
    elif inactive_mo <= 12:
        active_score = 0.20
    else:
        active_score = 0.08

    # Notice period
    notice = sig.get("notice_period_days", 90)
    if notice <= 0:
        notice_score = 1.00
    elif notice <= 15:
        notice_score = 0.95
    elif notice <= 30:
        notice_score = 0.85
    elif notice <= 60:
        notice_score = 0.65
    elif notice <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.25

    # Location
    loc = _norm(profile.get("location", ""))
    country = _norm(profile.get("country", ""))
    relocate = sig.get("willing_to_relocate", False)
    if any(pl in loc for pl in PREFERRED_LOCATIONS):
        loc_score = 1.00
    elif country == "india":
        loc_score = 0.80 if relocate else 0.55
    else:
        loc_score = 0.40 if relocate else 0.20

    # Active job-seeking signal
    apps = sig.get("applications_submitted_30d", 0)
    app_score = min(1.0, (apps + 1) / 6.0) if apps >= 0 else 0.5

    return _clamp(
        open_score   * 0.25
        + active_score * 0.40
        + notice_score * 0.20
        + loc_score    * 0.10
        + app_score    * 0.05
    )


# ─────────────────────────────────────────────────────────────────────────────
# Component 5 — Education  (weight 0.10)
# ─────────────────────────────────────────────────────────────────────────────

def _score_education(candidate: dict) -> float:
    """
    Evaluates best education entry by tier × field relevance × degree level.
    CS/AI/Stats fields get a 1.2× multiplier.
    Missing education gets a neutral 0.35.
    """
    edu_list: list[dict] = candidate.get("education", [])
    if not edu_list:
        return 0.35

    best = 0.0
    for edu in edu_list:
        tier_score = EDU_TIER.get(edu.get("tier", "unknown"), 0.40)
        field = _norm(edu.get("field_of_study", ""))
        degree = _norm(edu.get("degree", ""))

        field_mult = 1.20 if any(f in field for f in RELEVANT_FIELDS) else 0.85

        # Degree level bonus
        if any(d in degree for d in ["ph.d", "phd", "d.sc"]):
            deg_mult = 1.20
        elif any(d in degree for d in ["m.tech", "m.e.", "m.s.", "m.sc", "mba", "m.b.a"]):
            deg_mult = 1.10
        else:
            deg_mult = 1.00

        score = _clamp(tier_score * field_mult * deg_mult)
        if score > best:
            best = score

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Behavioural modifier  (multiplicative, range 0.30–1.20)
# ─────────────────────────────────────────────────────────────────────────────

def _behaviour_modifier(candidate: dict) -> float:
    """
    Uses the 23 redrob_signals to produce a multiplicative modifier.
    Signals are designed so that an unresponsive/inactive candidate
    is penalised even if their static profile is excellent.
    """
    sig = candidate.get("redrob_signals", {})

    # 1. Recruiter response rate (0-1, higher = better)
    response = sig.get("recruiter_response_rate", 0.5)
    response_score = response  # already [0,1]

    # 2. Interview completion rate
    interview = sig.get("interview_completion_rate", 0.5)
    interview_score = interview

    # 3. Offer acceptance rate (-1 = no history → neutral 0.5)
    offer_raw = sig.get("offer_acceptance_rate", -1)
    offer_score = 0.50 if offer_raw < 0 else float(offer_raw)

    # 4. Average response time (lower = better, cap at 48h)
    avg_rt = sig.get("avg_response_time_hours", 24.0)
    # Score: 0h=1.0, 12h=0.85, 24h=0.70, 48h=0.50, 120h=0.25, 200h+=0.10
    if avg_rt <= 4:
        rt_score = 1.00
    elif avg_rt <= 12:
        rt_score = 0.90
    elif avg_rt <= 24:
        rt_score = 0.75
    elif avg_rt <= 48:
        rt_score = 0.60
    elif avg_rt <= 96:
        rt_score = 0.40
    elif avg_rt <= 168:
        rt_score = 0.25
    else:
        rt_score = 0.12

    # 5. Recruiter engagement (search appearances + saves)
    search = sig.get("search_appearance_30d", 0)
    saves = sig.get("saved_by_recruiters_30d", 0)
    # Log-normalise, cap at 1.0
    engagement = _clamp(
        (math.log1p(search) / 7.5 + math.log1p(saves) / 4.5) / 2.0
    )

    # 6. GitHub activity score
    github_raw = sig.get("github_activity_score", -1)
    if github_raw < 0:
        github_score = 0.35  # no GitHub = mild negative for this role
    else:
        github_score = github_raw / 100.0

    # 7. Verification trust score
    verified = (
        (1 if sig.get("verified_email", False) else 0)
        + (1 if sig.get("verified_phone", False) else 0)
        + (1 if sig.get("linkedin_connected", False) else 0)
    ) / 3.0

    # 8. Profile completeness
    completeness = sig.get("profile_completeness_score", 50.0) / 100.0

    raw = (
        response_score  * 0.25
        + interview_score * 0.20
        + offer_score     * 0.08
        + rt_score        * 0.12
        + engagement      * 0.15
        + github_score    * 0.10
        + verified        * 0.05
        + completeness    * 0.05
    )
    # Map [0,1] → [0.30, 1.20]
    return _clamp(0.30 + raw * 0.90, 0.30, 1.20)


# ─────────────────────────────────────────────────────────────────────────────
# Honeypot score (forced low)
# ─────────────────────────────────────────────────────────────────────────────

_HONEYPOT_SCORE = 0.001


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning generator
# ─────────────────────────────────────────────────────────────────────────────

def _make_reasoning(
    candidate: dict,
    scores: dict[str, float],
    bmod: float,
    is_hp: bool,
) -> str:
    """
    Build a plain-language, candidate-specific reasoning string.
    Spec §3 penalises: empty, identical, templated, hallucinated, contradictory.
    We include candidate-specific facts only, no hallucination.
    """
    if is_hp:
        return "Honeypot flag: profile contains impossible timeline or fabricated signals; excluded."

    profile = candidate.get("profile", {})
    sig = candidate.get("redrob_signals", {})
    career: list[dict] = candidate.get("career_history", [])
    skills: list[dict] = candidate.get("skills", [])

    title = profile.get("current_title", "?")
    company = profile.get("current_company", "?")
    yoe = profile.get("years_of_experience", 0)
    loc = profile.get("location", "?")

    inactive_mo = round(_months_ago(sig.get("last_active_date")), 1)
    notice = sig.get("notice_period_days", "?")
    response = sig.get("recruiter_response_rate", 0)
    github = sig.get("github_activity_score", -1)
    open_wk = sig.get("open_to_work_flag", False)

    # Premium skills this candidate actually has
    premium_skills = [
        s["name"] for s in skills
        if _skill_tiers(s.get("name", ""))[1]
    ][:4]

    # Services career flag
    full_consulting = career and all(_is_services(r.get("company", "")) for r in career)

    # Build parts
    parts = []
    parts.append(f"{title} @ {company} ({yoe:.1f} yrs)")

    if premium_skills:
        parts.append(f"Premium AI skills: {', '.join(premium_skills)}")
    else:
        parts.append("No premium AI/ML skills detected")

    if full_consulting:
        parts.append("⚠ Full-consulting career (JD disqualifier)")

    parts.append(
        f"Behavioural: resp={response:.0%}, "
        f"inactive={inactive_mo}mo, notice={notice}d"
    )

    if open_wk:
        parts.append("Open to work ✓")
    else:
        parts.append("Not marked open-to-work")

    if github >= 0:
        parts.append(f"GitHub={github:.0f}/100")

    reason = " | ".join(parts)
    # Hard limit to avoid CSV parsing issues
    return reason[:350]


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {
    "skills":       0.35,
    "career":       0.25,
    "experience":   0.15,
    "availability": 0.15,
    "education":    0.10,
}


def score_candidate(candidate: dict) -> tuple[float, str]:
    """Return (final_score ∈ [0,1], reasoning_string)."""
    is_hp = _is_honeypot(candidate)

    if is_hp:
        reasoning = _make_reasoning(candidate, {}, 0.0, True)
        return _HONEYPOT_SCORE, reasoning

    s_skills = _score_skills(candidate)
    s_career = _score_career(candidate)
    s_exp    = _score_experience(candidate)
    s_avail  = _score_availability(candidate)
    s_edu    = _score_education(candidate)

    component = (
        s_skills * WEIGHTS["skills"]
        + s_career * WEIGHTS["career"]
        + s_exp    * WEIGHTS["experience"]
        + s_avail  * WEIGHTS["availability"]
        + s_edu    * WEIGHTS["education"]
    )

    bmod  = _behaviour_modifier(candidate)
    final = _clamp(component * bmod)

    scores = {
        "skills": s_skills,
        "career": s_career,
        "exp":    s_exp,
        "avail":  s_avail,
        "edu":    s_edu,
    }
    reasoning = _make_reasoning(candidate, scores, bmod, False)
    return final, reasoning


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Redrob AI Challenge — Candidate Ranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--candidates",
        default="./candidates.jsonl",
        help="Path to candidates.jsonl (one JSON object per line)",
    )
    parser.add_argument(
        "--out",
        default="./submission.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top candidates to output (spec requires exactly 100)",
    )
    args = parser.parse_args()

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        print(f"ERROR: candidates file not found: {candidates_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[rank.py] Loading {candidates_path} ...", flush=True)

    scored: list[tuple[float, str, str]] = []  # (score, cid, reasoning)
    skipped = 0

    with candidates_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                cand = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"  WARN line {lineno}: JSON parse error — {exc}",
                    file=sys.stderr,
                )
                skipped += 1
                continue

            cid = cand.get("candidate_id", f"UNKNOWN_{lineno}")
            final, reasoning = score_candidate(cand)
            scored.append((final, cid, reasoning))

            if lineno % 20000 == 0:
                print(f"  ... {lineno:,} candidates processed", flush=True)

    print(f"[rank.py] Total scored: {len(scored):,}  (skipped: {skipped})")

    # Sort: score descending, then candidate_id ascending for tie-break (per spec)
    # Round scores to 4 decimal places before sorting to match validation logic
    scored_rounded = [(round(s, 4), cid, reasoning) for s, cid, reasoning in scored]
    scored_rounded.sort(key=lambda x: (-x[0], x[1]))

    top_n = min(args.top_n, len(scored_rounded))
    top = scored_rounded[:top_n]

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, reasoning) in enumerate(top, start=1):
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    print(f"[rank.py] Submission written -> {out_path}  ({top_n} rows)")
    top5 = [f"{s[0]:.4f}" for s in top[:5]]
    print(f"[rank.py] Top-5 scores: {top5}")
    print(f"[rank.py] Score range: {top[-1][0]:.4f} - {top[0][0]:.4f}")


if __name__ == "__main__":
    main()
