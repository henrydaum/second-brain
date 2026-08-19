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
the prune ask the same object.

**A folder entry is a run of path segments, and being absolute only decides
where the run may start.** That one rule replaced two: "a bare name matches any
component" and "a full path matches by prefix". They were never different
rules — a bare name is the one-segment case — and keeping them apart cost real
behaviour. A person who typed ``Users/henry/My Drive/Photos`` without the
leading slash got an entry that could match nothing at all, because a relative
path has no base to resolve against and every indexed path is absolute. Now it
matches those segments wherever they run consecutively, which includes the
absolute place they meant. Nothing has to *guess* the missing root, which is
the part a coercion could not have done safely.

Segments also make the prefix bug structurally impossible: ``/vault/Archive``
compared as text claims ``/vault/Archived``, and comparing segment lists cannot.

Everything is normalized with the ``os.path`` helpers rather than by hand, so
one implementation is right on every platform: ``normcase`` lowercases on
Windows and is identity on POSIX, and ``altsep`` is ``/`` on Windows and
``None`` everywhere else.
"""


def normalize_extension(value) -> str:
	"""Return a comparable suffix (``.log``), or "" for an entry naming nothing.

	Comparable means: what ``Path.suffix.lower()`` answers for a file of that
	type, since that is the other side of every comparison. A leading ``*`` is
	dropped so the glob spelling people reach for first (``*.log``) means what
	it looks like it means.
	"""
	text = str(value).strip().lower().lstrip("*")
	if not text:
		return ""
	return text if text.startswith(".") else "." + text


def _respell(text: str) -> str:
	r"""Rewrite a hand-typed path in this platform's spelling.

	Both separators are accepted on either platform, repeats are collapsed, and
	a trailing separator is dropped, so ``\Photos\Misc\``, ``/Photos//Misc``
	and ``Photos/Misc`` all arrive as one thing. Accepting the foreign
	separator costs the ability to name a POSIX folder containing a literal
	backslash, which is a legal filename character and not a thing anybody
	types into a settings field on purpose.
	"""
	unified = text.strip().replace("\\", os.sep).replace("/", os.sep)
	if not unified:
		return ""
	respelled = os.path.normpath(unified)
	# POSIX normpath keeps exactly two leading slashes, which is a real
	# distinction in the standard and never one an indexed path carries.
	# Windows keeps them too, and there it is a UNC host that must survive.
	if os.altsep is None and respelled.startswith(os.sep * 2):
		respelled = os.sep + respelled.lstrip(os.sep)
	return respelled


def _segments(value) -> tuple:
	"""The comparable parts of a path: native separators, native case."""
	return tuple(os.path.normcase(part) for part in Path(value).parts)


@dataclass(frozen=True)
class _FolderRule:
	"""One ``ignored_folders`` entry, as segments plus where they may start."""

	segments: tuple
	anchored: bool

	def matches(self, parts: tuple) -> bool:
		"""Whether these segments run consecutively inside ``parts``."""
		width = len(self.segments)
		if not width or width > len(parts):
			return False
		if self.anchored:
			return parts[:width] == self.segments
		return any(parts[i:i + width] == self.segments
		           for i in range(len(parts) - width + 1))

	@classmethod
	def parse(cls, entry) -> "_FolderRule | None":
		"""Build a rule from one raw entry, or None when it names no folder.

		A root on its own is dropped rather than honoured. ``/`` respells to a
		single root segment, which as an ordinary rule matches the root segment
		every absolute path starts with — so one stray character in the
		settings field would silently exclude the entire index. It is never a
		useful thing to mean, and it is the one entry whose mistake is total.
		"""
		respelled = _respell(str(entry))
		if not respelled or respelled == os.curdir:
			return None
		if not os.path.splitdrive(respelled)[1].strip(os.sep):
			return None
		return cls(segments=_segments(respelled),
		           anchored=os.path.isabs(respelled))


@dataclass(frozen=True)
class IgnoreRules:
	"""Whether a file may be in the index, from the settings that decide it.

	Frozen because the watcher swaps in a whole new one on ``rescan`` rather
	than mutating fields in place — the old shape had a scan thread reading
	them while a config write updated them one at a time.
	"""

	folders: tuple
	extensions: frozenset
	skip_hidden: bool

	@classmethod
	def from_config(cls, config: dict) -> "IgnoreRules":
		"""Build the rules from a config dict, normalizing every entry."""
		folders = []
		for entry in config.get("ignored_folders") or []:
			rule = _FolderRule.parse(entry)
			if rule is not None and rule not in folders:
				folders.append(rule)

		extensions = {
			normalize_extension(entry)
			for entry in (config.get("ignored_extensions") or [])
		}
		extensions.discard("")

		return cls(
			folders=tuple(folders),
			extensions=frozenset(extensions),
			skip_hidden=bool(config.get("skip_hidden_folders", True)),
		)

	def ignores_folder(self, path) -> bool:
		"""Whether a directory is excluded by an entry or for being hidden."""
		parts = _segments(path)

		if any(rule.matches(parts) for rule in self.folders):
			return True

		# "." and ".." are relative-path punctuation, not hidden directories.
		# Path(".").parts is (".",), so a bare filename's parent would
		# otherwise exclude itself.
		if self.skip_hidden and any(
				part.startswith(".") and part not in (".", "..") for part in parts):
			return True

		return False

	def ignores_extension(self, extension) -> bool:
		"""Whether an extension is excluded. Accepts ``pdf``, ``.pdf`` or ``.PDF``."""
		return normalize_extension(extension) in self.extensions

	def excludes(self, path) -> bool:
		"""Whether an indexed file should be dropped under the current rules.

		The folder rules are asked about the *parent*, deliberately: applied to
		the whole path they would read the file's own name as a segment too, so
		a file called ``node_modules`` would be excluded by the defaults.
		"""
		p = Path(path)
		return self.ignores_folder(str(p.parent)) or self.ignores_extension(p.suffix)
