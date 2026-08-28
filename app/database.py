# here we will use sqlite3 for monitoring nd storing the predictions and everything 

import sqlite3
import datetime

# the following function creates the database nd table if they dont exist
def init_db():
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        prediction TEXT,
        confidence REAL,
        latency_ms REAL,
        input_method TEXT
    )''')

    conn.commit()
    conn.close()

def log_prediction(prediction, confidence, latency_ms, input_method):
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions 
        (timestamp, prediction, confidence, latency_ms, input_method)
        VALUES (?, ?, ?, ?, ?)''',
        (datetime.datetime.now().isoformat(),
         prediction, confidence, latency_ms, input_method))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect('monitoring.db')
    c = conn.cursor()
    c.execute('SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100')
    rows = c.fetchall()
    conn.close()
    return rows