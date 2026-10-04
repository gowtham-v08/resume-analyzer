"""
Resume Analyzer (ATS Score Checker)
Scores how well a resume matches a job description using TF-IDF + cosine similarity.
"""

import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

st.set_page_config(
    page_title="Resume Analyzer | ATS Score Checker",
    page_icon="📄",
    layout="wide",
)

st.title("📄 Resume Analyzer (ATS Score Checker)")
st.caption("Compare a resume against a job description using TF-IDF and cosine similarity.")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text: str, top_n: int = 25) -> list[str]:
    words = clean_text(text).split()
    # Prefer longer technical-looking tokens
    scored = {}
    for w in words:
        if len(w) < 3:
            continue
        scored[w] = scored.get(w, 0) + 1
    ranked = sorted(scored.items(), key=lambda x: (-x[1], -len(x[0]), x[0]))
    return [w for w, _ in ranked[:top_n]]


def score_match(resume: str, jd: str) -> tuple[float, list[str], list[str]]:
    resume_c = clean_text(resume)
    jd_c = clean_text(jd)
    if not resume_c or not jd_c:
        return 0.0, [], []

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([resume_c, jd_c])
    sim = float(cosine_similarity(matrix[0:1], matrix[1:2])[0][0])
    score = round(sim * 100, 1)

    jd_kw = set(extract_keywords(jd, 30))
    resume_kw = set(extract_keywords(resume, 40))
    matched = sorted(jd_kw & resume_kw)
    missing = sorted(jd_kw - resume_kw)
    return score, matched, missing


col1, col2 = st.columns(2)

with col1:
    st.subheader("Resume")
    resume_text = st.text_area(
        "Paste resume text",
        height=280,
        placeholder="Paste the full resume text here...",
        label_visibility="collapsed",
    )

with col2:
    st.subheader("Job Description")
    jd_text = st.text_area(
        "Paste job description",
        height=280,
        placeholder="Paste the job description here...",
        label_visibility="collapsed",
    )

if st.button("Analyze Match", type="primary", use_container_width=True):
    if not resume_text.strip() or not jd_text.strip():
        st.warning("Please paste both resume and job description text.")
    else:
        score, matched, missing = score_match(resume_text, jd_text)

        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("ATS Match Score", f"{score}%")
        m2.metric("Matched Keywords", len(matched))
        m3.metric("Missing Keywords", len(missing))

        if score >= 70:
            st.success("Strong match — resume is well aligned with the JD.")
        elif score >= 45:
            st.info("Moderate match — add the missing skills below to improve score.")
        else:
            st.error("Low match — consider rewriting the resume for this role.")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("✅ Matched keywords")
            st.write(", ".join(matched) if matched else "None detected")
        with c2:
            st.subheader("❌ Missing keywords (improve these)")
            st.write(", ".join(missing) if missing else "None — great coverage!")

st.markdown("---")
st.caption("Built with Python · scikit-learn · Streamlit  |  github.com/gowtham-v08/resume-analyzer")
