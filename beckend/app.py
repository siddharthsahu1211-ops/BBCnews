from flask import Flask
from routes import bp
from db import get_conn

def ensure_schema():
    """Create database tables if they don't exist"""
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        # Create articles table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id SERIAL PRIMARY KEY,
                category VARCHAR(50) NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create index on category for faster queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_articles_category 
            ON articles(category);
        """)
        
        conn.commit()
        print("✅ Database schema created successfully")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Schema creation error: {e}")

def create_app():
    app=Flask(__name__)
    app.register_blueprint(bp)
    return app

app=create_app()

if __name__=="__main__":
    ensure_schema()
    app.run(host="0.0.0.0",port=5000,debug=True)


