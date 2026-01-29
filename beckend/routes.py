from flask import Blueprint, jsonify
import json
from db import get_conn

#g
bp = Blueprint('routes', __name__)

@bp.get("/health")
def health():
    return{"status":"ok"}

@bp.get("/articles")
def get_articles():
    """Get all articles, optionally filtered by category"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT id, category, title, content, created_at FROM articles LIMIT 100")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        articles = [
            {
                "id": row[0],
                "category": row[1],
                "title": row[2],
                "content": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                "created_at": str(row[4])
            }
            for row in rows
        ]
        
        return jsonify({"status": "success", "count": len(articles), "articles": articles})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@bp.get("/articles/<category>")
def get_articles_by_category(category):
    """Get articles by category"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT id, category, title, content, created_at FROM articles WHERE category = %s LIMIT 50",
            (category,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        articles = [
            {
                "id": row[0],
                "category": row[1],
                "title": row[2],
                "content": row[3][:200] + "..." if len(row[3]) > 200 else row[3],
                "created_at": str(row[4])
            }
            for row in rows
        ]
        
        return jsonify({"status": "success", "category": category, "count": len(articles), "articles": articles})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500