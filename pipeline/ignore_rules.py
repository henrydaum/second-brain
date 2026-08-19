"""Pipeline support for ignore rules."""

import os
from dataclasses import dataclass
from pathlib import Path

"""
Ignore rules.

The two settings that decide what may enter the index — ``ignored_folders``
and ``ignored_extensions`` — are free-text JSON lists, and nothing between the
settings panel and the walk ever normalized them. ``config_manager._normalize_
list`` coerces the value to a list and never touches the items, so ``log`` and
``.LOG`` were both compared against ``Path.suffix.lower()`` — which always
carries a dot and is always lowercase — and matched nothing, while a folder
entered as a full path was compared against individual path *components* and
matched nothing either.

Every one of those failures is silent, which is what made the bug expensive:
the write succeeds, the rescan fires, ``/config`` shows the setting as set, and
no file is filtered. So normalization happens once, here, and both the walk and
the prune ask the same object. It is a module rather than methods on ``Watcher``
because ``__init__`` and ``rescan`` each used to take their own snapshot of the
same four keys, which is two places to keep in step.

Everything is normalized with the ``os.path`` helpers rather than by hand, so
one implementation is right on every platform: ``normcase`` lowercases on
Windows and is identity on POSIX, and ``altsep`` is ``/`` on Windows and
``None`` everywhere else.
"""


def normalize_extension(value) -> str:
	"""Return a comparable suffix (``.log``), or "" for an entry naming nothing.

	Comparable means: what ``Path.suffix.lower()`` answers for a file of that
	type, since that is the other side of every comparison.
	"""
	text = str(value).strip().lower()
	if not text:
		return ""
	return text if text.startswith(".") else "." + text


def _is_path_like(text: str) -> bool:
	"""Whether an ``ignored_folders`` entry names a location rather than a name."""
	if os.path.isabs(text):
		return True
	if os.sep in text:
		return True
	return bool(os.altsep) and os.altsep in text


def _normalize_path(value) -> str:
	"""Return a path in the form every comparison uses: native separators, native case."""
	return os.path.normcase(os.path.normpath(str(value)))


@dataclass(frozen=True)
class IgnoreRules:
	"""Whether a file may be in the index, from the four settings that decide it.

	Frozen because the watcher swaps in a whole new one on ``rescan`` rather
	than mutating four attributes in place — the old shape had a scan thread
	reading the fields while a config write updated them one at a time.
	"""

	folder_names: frozenset
	folder_paths: tuple
	extensions: frozenset
	skip_hidden: bool

	@classmethod
	def from_config(cls, config: dict) -> "IgnoreRules":
		"""Build the rules from a config dict, normalizing every entry."""
		names = set()
		paths = []
		for entry in config.get("ignored_folders") or []:
			text = str(entry).strip()
			if not text:
				continue
			if _is_path_like(text):
				paths.append(_normalize_path(text))
			else:
				names.add(os.path.normcase(text))

		extensions = {
			normalize_extension(entry)
			for entry in (config.get("ignored_extensions") or [])
		}
		extensions.discard("")

		return cls(
			folder_names=frozenset(names),
			folder_paths=tuple(dict.fromkeys(paths)),
			extensions=frozenset(extensions),
			skip_hidden=bool(config.get("skip_hidden_folders", True)),
		)

	def ignores_folder(self, path) -> bool:
		"""Whether a directory is excluded by name, by location, or for being hidden."""
		parts = Path(path).parts

		if self.folder_names and any(
				os.path.normcase(part) in self.folder_names for part in parts):
			return True

		# "." and ".." are relative-path punctuation, not hidden directories.
		# Path(".").parts is (".",), so a bare filename's parent would
		# otherwise exclude itself.
		if self.skip_hidden and any(
				part.startswith(".") and part not in (".", "..") for part in parts):
			return True

		if self.folder_paths:
			candidate = _normalize_path(path)
			for prefix in self.folder_paths:
				# The separator is what stops "/x/Archive" also claiming
				# "/x/Archived".
				if candidate == prefix or candidate.startswith(prefix + os.sep):
					return True

		return False

	def ignores_extension(self, extension) -> bool:
		"""Whether an extension is excluded. Accepts ``pdf``, ``.pdf`` or ``.PDF``."""
		return normalize_extension(extension) in self.extensions

	def excludes(self, path) -> bool:
		"""Whether an indexed file should be dropped under the current rules.

		The folder rules are asked about the *parent*, deliberately: applied to
		the whole path they would read the file's own name as a component too,
		so a file called ``node_modules`` would be excluded by the defaults.
		"""
		p = Path(path)
		return self.ignores_folder(str(p.parent)) or self.ignores_extension(p.suffix)
