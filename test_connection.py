#!/usr/bin/env python3
import sys
sys.path.insert(0, '/workspaces/BBCnews/beckend')

from db import get_conn

try:
    conn = get_conn()
    print("✓ Connection to Supabase successful!")
    conn.close()
except Exception as e:
    print(f"✗ Connection failed: {e}")
    sys.exit(1)
