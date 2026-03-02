import logging
import sqlite3
import threading

logger = logging.getLogger(__name__)

class Database:
	def __init__(self, db_path: Path):
		self.db_path = db_path
		# Allow multiple threads to use this connection (with locking)
		self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
		self.lock = threading.Lock() # Application-level lock for safety

		self._setup_tables()

	def _setup_tables(self):
		# WAL (write-ahead logging) mode - read and write can occur simultaneously
		self.conn.execute("PRAGMA journal_mode=WAL;")

		# Primary Key is (path, task_type)
		self.conn.execute("""
			CREATE TABLE IF NOT EXISTS tasks (
				path TEXT,
				task_type TEXT,
				status TEXT DEFAULT 'PENDING',
				file_mtime REAL,
				PRIMARY KEY(path, task_type)
			)
		""")
		
		