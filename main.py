from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from sqlalchemy import create_engine, text

app = FastAPI()

model = joblib.load("model.pkl")

DATABASE_URL = "postgresql://localhost/ml_api_db"
engine = create_engine(DATABASE_URL)

KEYWORDS = {"ai", "tech", "market", "economy", "startup"}

class ArticleInput(BaseModel):
    title: str
    content: str
    is_recent: int

def count_keywords(text):
    words = text.lower().split()
    return sum(1 for w in words if w.strip(".,!?") in KEYWORDS)

def extract_features(article):
    text = f"{article.title} {article.content}"
    article_length = len(text.split())
    keyword_count = count_keywords(text)
    return [article_length, keyword_count, article.is_recent]

@app.get("/")
def home():
    return {"message": "News Ranker API running"}

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "database connected"}
    except Exception as e:
        return {"status": "error", "details": str(e)}

@app.post("/rank")
def rank_article(article: ArticleInput):
    features = extract_features(article)
    score = float(model.predict_proba([features])[0][1])

    with engine.connect() as conn:
        conn.execute(
            text("""
            INSERT INTO articles (title, content, score, length, keywords, recent)
            VALUES (:title, :content, :score, :length, :keywords, :recent)
            """),
            {
                "title": article.title,
                "content": article.content,
                "score": score,
                "length": features[0],
                "keywords": features[1],
                "recent": features[2]
            }
        )
        conn.commit()

    return {
        "title": article.title,
        "score": score,
        "features": {
            "length": features[0],
            "keywords": features[1],
            "recent": features[2]
        }
    }

@app.get("/articles")
def get_articles():
    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT id, title, content, score, length, keywords, recent, created_at
            FROM articles
            ORDER BY created_at DESC
            """)
        )
        rows = result.mappings().all()

    return {"articles": [dict(row) for row in rows]}

@app.get("/articles/{article_id}")
def get_article(article_id: int):
    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT id, title, content, score, length, keywords, recent, created_at
            FROM articles
            WHERE id = :article_id
            """),
            {"article_id": article_id}
        )
        row = result.mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="Article not found")

    return dict(row)