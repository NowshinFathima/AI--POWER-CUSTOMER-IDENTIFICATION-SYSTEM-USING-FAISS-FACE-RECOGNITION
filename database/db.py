import sqlite3
import numpy as np

DB_NAME = "customers.db"

def get_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    conn.close()

def insert_customer(name, embedding):
    conn = get_connection()
    conn.execute(
        "INSERT INTO customers (name, embedding) VALUES (?, ?)",
        (name, embedding.tobytes())
    )
    conn.commit()
    conn.close()

def fetch_customers():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, name, embedding FROM customers")
    rows = cur.fetchall()
    conn.close()

    data = []
    for r in rows:
        emb = np.frombuffer(r[2], dtype=np.float32)
        data.append((r[0], r[1], emb))
    return data
