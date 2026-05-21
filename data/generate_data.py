import numpy as np
import pandas as pd
import json
import os

SKILLS = [
    'Python', 'JavaScript', 'TypeScript', 'Java', 'Go', 'SQL', 'C++', 'Rust',
    'React', 'FastAPI', 'Django', 'Next.js', 'Node.js', 'Vue.js',
    'TensorFlow', 'PyTorch', 'scikit-learn', 'pandas', 'NumPy',
    'AWS', 'GCP', 'Azure', 'Docker', 'Kubernetes', 'Terraform',
    'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision', 'LLMs', 'RAG',
    'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch',
    'Spark', 'dbt', 'Airflow', 'Kafka',
]

ROLES = [
    'Software Engineer', 'Senior Software Engineer', 'Staff Engineer',
    'Data Scientist', 'Senior Data Scientist', 'ML Engineer', 'Senior ML Engineer',
    'Frontend Engineer', 'Backend Engineer', 'Full Stack Engineer',
    'Data Engineer', 'DevOps Engineer', 'Platform Engineer',
]

EDUCATION = ["High School", "Associate's", "Bachelor's", "Master's", "PhD"]
EDU_LEVEL = {e: i for i, e in enumerate(EDUCATION)}

ROLE_SKILLS = {
    'Software Engineer':        ['Python', 'JavaScript', 'SQL', 'Docker'],
    'Senior Software Engineer': ['Python', 'TypeScript', 'SQL', 'AWS', 'Docker'],
    'Staff Engineer':           ['Python', 'Go', 'AWS', 'Kubernetes', 'PostgreSQL'],
    'Data Scientist':           ['Python', 'pandas', 'scikit-learn', 'SQL', 'TensorFlow'],
    'Senior Data Scientist':    ['Python', 'TensorFlow', 'PyTorch', 'SQL', 'AWS', 'Machine Learning'],
    'ML Engineer':              ['Python', 'TensorFlow', 'PyTorch', 'Docker', 'AWS', 'Machine Learning'],
    'Senior ML Engineer':       ['Python', 'TensorFlow', 'PyTorch', 'Kubernetes', 'AWS', 'LLMs', 'RAG'],
    'Frontend Engineer':        ['JavaScript', 'TypeScript', 'React', 'Next.js'],
    'Backend Engineer':         ['Python', 'Java', 'Go', 'PostgreSQL', 'Redis', 'Docker'],
    'Full Stack Engineer':      ['Python', 'JavaScript', 'React', 'PostgreSQL', 'Docker'],
    'Data Engineer':            ['Python', 'SQL', 'Spark', 'Airflow', 'dbt', 'AWS'],
    'DevOps Engineer':          ['Docker', 'Kubernetes', 'AWS', 'Terraform', 'Python'],
    'Platform Engineer':        ['Go', 'Kubernetes', 'AWS', 'Terraform', 'Python'],
}

SKILL_TRENDS = {
    'Python':           (7.5, 0.08),
    'JavaScript':       (8.0, 0.04),
    'TypeScript':       (5.0, 0.18),
    'Java':             (7.0, -0.03),
    'Go':               (4.0, 0.15),
    'SQL':              (8.0, 0.02),
    'React':            (7.0, 0.06),
    'TensorFlow':       (4.5, 0.12),
    'PyTorch':          (3.5, 0.20),
    'AWS':              (7.5, 0.07),
    'Docker':           (6.0, 0.10),
    'Kubernetes':       (5.0, 0.14),
    'Machine Learning': (5.0, 0.15),
    'LLMs':             (1.0, 0.80),
    'RAG':              (0.5, 1.20),
    'Rust':             (2.0, 0.25),
    'Spark':            (5.5, 0.03),
    'dbt':              (2.0, 0.35),
}


def generate_match_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for _ in range(n):
        num_skills   = rng.integers(4, 16)
        resume_skills = set(rng.choice(SKILLS, size=num_skills, replace=False).tolist())
        resume_exp   = int(rng.integers(0, 16))
        resume_edu   = rng.choice(EDUCATION, p=[0.05, 0.05, 0.55, 0.30, 0.05])

        role           = rng.choice(ROLES)
        base_skills    = set(ROLE_SKILLS[role])
        extra_required = set(rng.choice(SKILLS, size=int(rng.integers(0, 4)), replace=False).tolist())
        required_skills = base_skills | extra_required

        req_exp = int(rng.integers(1, 9))
        req_edu = rng.choice(EDUCATION[:4], p=[0.02, 0.03, 0.60, 0.35])

        overlap   = len(resume_skills & required_skills) / len(required_skills)
        missing   = len(required_skills - resume_skills)
        extra     = len(resume_skills - required_skills)
        exp_ratio = min(resume_exp / max(req_exp, 1), 2.0)
        edu_match = 1 if EDU_LEVEL[resume_edu] >= EDU_LEVEL[req_edu] else 0

        score = (
            overlap   * 0.55 +
            min(exp_ratio, 1.0) * 0.30 +
            edu_match * 0.15
        )
        score = float(np.clip(score + rng.normal(0, 0.03), 0, 1))

        records.append({
            'skill_overlap':   overlap,
            'missing_skills':  missing,
            'extra_skills':    min(extra, 10),
            'exp_ratio':       exp_ratio,
            'edu_match':       edu_match,
            'req_exp':         req_exp,
            'resume_exp':      resume_exp,
            'role_idx':        ROLES.index(role),
            'match_score':     score,
            'resume_skills':   ','.join(sorted(resume_skills)),
            'required_skills': ','.join(sorted(required_skills)),
            'missing_list':    ','.join(sorted(required_skills - resume_skills)),
            'role':            role,
        })

    return pd.DataFrame(records)


def generate_trend_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quarters = [
        (y, q) for y in range(2020, 2027)
        for q in range(1, 5)
        if not (y == 2026 and q > 2)
    ]
    records = []

    for skill, (base_demand, growth) in SKILL_TRENDS.items():
        skill_idx = list(SKILL_TRENDS.keys()).index(skill)
        for i, (year, quarter) in enumerate(quarters):
            t      = i / 4.0
            demand = float(np.clip(base_demand * (1 + growth) ** t + rng.normal(0, 0.2), 0, 10))
            records.append({
                'skill':        skill,
                'skill_idx':    skill_idx,
                'year':         year,
                'quarter':      quarter,
                'time_idx':     i,
                'demand_score': round(demand, 2),
            })

    return pd.DataFrame(records)


if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)

    match_df = generate_match_data(6000)
    match_df.to_csv('data/match_data.csv', index=False)
    print(f"Match data: {len(match_df)} rows, avg score: {match_df['match_score'].mean():.2f}")

    trend_df = generate_trend_data()
    trend_df.to_csv('data/trend_data.csv', index=False)
    print(f"Trend data: {len(trend_df)} rows")

    meta = {'skills': list(SKILL_TRENDS.keys()), 'roles': ROLES,
            'all_skills': SKILLS, 'education': EDUCATION}
    with open('data/meta.json', 'w') as f:
        json.dump(meta, f)
    print("Metadata saved.")
