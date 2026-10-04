# Resume Analyzer (ATS Score Checker)

Web app that scores how well a **resume** matches a **job description** using **TF-IDF** and **cosine similarity**.

## Live Demo

Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud) (free):

1. Fork / open this repo  
2. New app → select `app.py`  
3. Deploy → get a public URL like `https://resume-analyzer-xxxx.streamlit.app`

## Features

- Paste resume + job description
- ATS-style match score (0–100%)
- Matched vs missing keywords
- Simple, recruiter-friendly UI

## Tech Stack

- Python
- scikit-learn (TF-IDF, cosine similarity)
- Streamlit

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project structure

```
resume-analyzer/
├── app.py
├── requirements.txt
└── README.md
```

## Author

**Gowtham V** · [github.com/gowtham-v08](https://github.com/gowtham-v08) · [Portfolio](https://gowtham-v.vercel.app)
