"""CanvasRuntime — the parallel-to-ConversationRuntime entry point.

Holds an in-memory registry of ``canvas_id -> CanvasState`` and exposes a
single ``handle_action(canvas_id, action_type, payload)`` method that wraps
the one labeled ``cs.enact(...)`` site for canvas dispatch.

When constructed with a ``db``, the runtime autosaves to the
``canvas_states`` table after every successful action and lazy-loads on
``get`` for canvases it hasn't seen yet. Without a ``db`` the runtime is
purely in-memory — handy for unit tests and stateless callers.

Zero coupling to ConversationRuntime. Mapping user/session -> canvas_id
lives outside; this module just owns the canvases.
"""

from __future__ import annotations

import logging
from typing import Any

from canvas import persistence as canvas_persistence
from canvas.canvas import Canvas
from canvas.state import CanvasState
from state_machine.errors import ActionResult, ERROR_INVALID_ACTION

logger = logging.getLogger("CanvasRuntime")


class CanvasRuntime:
	"""In-memory registry of CanvasStates + dispatch + optional persistence."""

	def __init__(self, db=None):
		"""Initialize the runtime.

		``db`` is an optional ``pipeline.database.Database`` instance.
		When provided, every successful action autosaves and ``get`` will
		lazy-load from ``canvas_states`` for unknown ids.
		"""
		self.canvases: dict[str, CanvasState] = {}
		# session_key -> canvas_id mapping. In-memory; resets on restart.
		# The canvases themselves survive via canvas_states (when db is set).
		self._session_to_canvas: dict[str, str] = {}
		self.db = db

	# ── lifecycle ────────────────────────────────────────────────────────

	def create_canvas(self, canvas_id: str | None = None) -> str:
		"""Allocate a fresh CanvasState, register it, and persist (if db)."""
		cs = CanvasState(canvas_id=canvas_id)
		self.canvases[cs.canvas_id] = cs
		self._persist(cs)
		logger.info("create_canvas id=%s", cs.canvas_id)
		return cs.canvas_id

	def register(self, cs: CanvasState) -> str:
		"""Insert an existing CanvasState (e.g. one loaded from a dict)."""
		self.canvases[cs.canvas_id] = cs
		self._persist(cs)
		return cs.canvas_id

	def get(self, canvas_id: str) -> CanvasState | None:
		"""Return the CanvasState for ``canvas_id``, lazy-loading from db if needed."""
		cs = self.canvases.get(canvas_id)
		if cs is not None:
			return cs
		if self.db is None:
			return None
		loaded = canvas_persistence.load(self.db, canvas_id)
		if loaded is None:
			return None
		self.canvases[loaded.canvas_id] = loaded
		return loaded

	def delete(self, canvas_id: str) -> None:
		"""Drop from registry and persistence."""
		self.canvases.pop(canvas_id, None)
		if self.db is not None:
			try:
				canvas_persistence.delete(self.db, canvas_id)
			except Exception:
				logger.exception("delete persistence failed for id=%s", canvas_id)
		logger.info("delete_canvas id=%s", canvas_id)

	def snapshot(self, canvas_id: str) -> dict | None:
		"""Serialize one canvas to a dict, or None if unknown."""
		cs = self.get(canvas_id)
		return cs.to_dict() if cs else None

	def remix(self, pool_hash: str) -> CanvasState | None:
		"""Materialize a fresh canvas from a pool_hash and register it.

		Looks up ``canvas_pools`` for the state at that hash, builds a
		brand-new ``CanvasState`` with its own canvas_id (private editing
		handle), and returns it. The caller is responsible for binding it
		to a session via ``bind_session``. Returns None if the pool_hash
		is unknown or the db isn't wired.
		"""
		if self.db is None:
			return None
		state = canvas_persistence.load_pool(self.db, pool_hash)
		if state is None:
			return None
		# Fresh editing handle; same content.
		cs = CanvasState(canvas=Canvas.from_dict(state))
		self.canvases[cs.canvas_id] = cs
		self._persist(cs)
		logger.info("remix pool=%s -> new canvas_id=%s", pool_hash, cs.canvas_id)
		return cs

	def for_session(self, session_key: str) -> CanvasState:
		"""Return the canvas bound to ``session_key``, creating it lazily.

		The session→canvas binding lives in-memory on the runtime. The
		canvas itself persists via canvas_states (if db is wired), so the
		state survives a restart even though the binding doesn't — a
		future "rebind on session resume" layer can rebuild the link.
		"""
		bound = self._session_to_canvas.get(session_key)
		if bound is not None:
			cs = self.get(bound)
			if cs is not None:
				return cs
			# Binding pointed at a deleted canvas; fall through and remint.
			self._session_to_canvas.pop(session_key, None)
		cid = self.create_canvas()
		self._session_to_canvas[session_key] = cid
		logger.info("for_session bound session=%s -> canvas=%s", session_key, cid)
		return self.canvases[cid]

	def current_for_session(self, session_key: str) -> CanvasState | None:
		"""Return the canvas bound to ``session_key`` without creating one."""
		bound = self._session_to_canvas.get(session_key)
		return self.get(bound) if bound else None

	def bind_session(self, session_key: str, canvas_id: str) -> None:
		"""Explicitly tie a session to an existing canvas (e.g. on share-link open)."""
		cs = self.get(canvas_id)
		if cs is None:
			raise KeyError(f"unknown canvas: {canvas_id!r}")
		self._session_to_canvas[session_key] = canvas_id

	def unbind_session(self, session_key: str) -> None:
		"""Forget the session→canvas mapping. The canvas itself is untouched."""
		self._session_to_canvas.pop(session_key, None)

	def list_ids(self) -> list[str]:
		"""Known canvas ids — union of in-memory + persisted."""
		ids = set(self.canvases.keys())
		if self.db is not None:
			try:
				ids.update(canvas_persistence.list_ids(self.db))
			except Exception:
				logger.exception("list_ids persistence query failed")
		return sorted(ids)

	# ── dispatch ─────────────────────────────────────────────────────────

	def handle_action(
		self,
		canvas_id: str,
		action_type: str,
		payload: Any = None,
	) -> ActionResult:
		"""Route ``action_type`` to the canvas at ``canvas_id`` and autosave on success."""
		cs = self.get(canvas_id)
		if cs is None:
			return ActionResult.fail(
				action_type, f"unknown canvas: {canvas_id!r}", code=ERROR_INVALID_ACTION,
			)
		# ──────────── THE enact() SITE (canvas-side) ────────────────
		result = cs.enact(action_type, payload)
		# ────────────────────────────────────────────────────────────
		if result.ok:
			self._persist(cs)
		return result

	# ── internal ─────────────────────────────────────────────────────────

	def _persist(self, cs: CanvasState) -> None:
		"""Save to db if configured. Log-but-don't-raise so a save failure
		doesn't corrupt the in-memory action result."""
		if self.db is None:
			return
		try:
			canvas_persistence.save(self.db, cs)
		except Exception:
			logger.exception("autosave failed for id=%s", cs.canvas_id)
