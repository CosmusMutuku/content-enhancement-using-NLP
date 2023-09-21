# database.py
import sqlite3

# Create a new SQLite database and connect to it
conn = sqlite3.connect('educational_content.db')

# Create a table to store user inputs and enhanced content
conn.execute('''CREATE TABLE IF NOT EXISTS content (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original TEXT,
    cleaned TEXT,
    sentiment_scores TEXT,
    sentiment_label TEXT,
    readability_grade REAL,
    summary TEXT
)''')

# Commit changes and close the connection
conn.commit()
conn.close()
