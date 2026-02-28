import sqlite3
import time
from pathlib import Path
from config import DB_PATH

class FeedbackAgent:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS feedback (
                        ts REAL,
                        id TEXT,
                        score REAL,
                        decision TEXT,
                        outcome INTEGER)''')
        con.commit()
        con.close()

    def log(self, id, score, decision, outcome=None):
        con = sqlite3.connect(self.db_path)
        cur = con.cursor()
        cur.execute('INSERT INTO feedback VALUES (?,?,?,?,?)',
                    (time.time(), id, score, decision, outcome if outcome is not None else -1))
        con.commit()
        con.close()
