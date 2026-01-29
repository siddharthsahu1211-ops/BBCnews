#!/usr/bin/env python3
"""
Populate Supabase with BBC news articles from the local dataset
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, '/workspaces/BBCnews/beckend')

from db import get_conn

def load_articles():
    """Load articles from bbc directory into Supabase"""
    bbc_dir = Path('/workspaces/BBCnews/bbc')
    
    if not bbc_dir.exists():
        print("❌ BBC directory not found")
        return
    
    try:
        conn = get_conn()
        cur = conn.cursor()
        
        categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
        total_articles = 0
        
        for category in categories:
            category_dir = bbc_dir / category
            
            if not category_dir.exists():
                print(f"⚠️ Category directory not found: {category}")
                continue
            
            # Get all .txt files in the category
            files = sorted(category_dir.glob('*.txt'))
            
            for file_path in files:
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # Extract title (first line) and content
                    lines = content.strip().split('\n')
                    title = lines[0] if lines else "Untitled"
                    article_content = '\n'.join(lines[1:]) if len(lines) > 1 else content
                    
                    # Insert into database
                    cur.execute("""
                        INSERT INTO articles (category, title, content)
                        VALUES (%s, %s, %s)
                        ON CONFLICT DO NOTHING;
                    """, (category, title, article_content))
                    
                    total_articles += 1
                    
                except Exception as e:
                    print(f"⚠️ Error reading {file_path}: {e}")
            
            print(f"✅ Loaded {category} articles")
        
        conn.commit()
        print(f"\n✅ Total articles loaded: {total_articles}")
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Loading BBC news articles into Supabase...")
    load_articles()
