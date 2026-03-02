import logging
from pathlib import Path
import os
import time

import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
from threading import BoundedSemaphore

logger = logging.getLogger(__name__)

class Orchestrator:
	def __init__(self, db, config, tasks):
		self.db = db
		self.config = config
		self.tasks = tasks

		self.queue = queue.PriorityQueue()
		self.executor = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4), thread_name_prefix="Worker")
		self.pool_semaphore = BoundedSemaphore(value=self.config.get('max_workers', 4))