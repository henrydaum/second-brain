"""Base class for canvas skills.

A skill is a Python file under plugins/skills/ (baked-in) or
DATA_DIR/sandbox_skills/ (agent/user authored) that defines a class
subclassing BaseSkill. The class declares metadata as class attributes and
implements `run(canvas, **params)` to either create or transform an image.

Skill code is executed in a subprocess sandbox (plugins/skills/helpers/
skill_sandbox_entry.py) with restricted imports and resource limits. The
sandbox imports the file, finds the BaseSkill subclass, instantiates it,
and calls instance.run(canvas, **params).

Allowed imports inside a skill: math, random, colorsys, numpy, PIL.*, and
``from plugins.BaseSkill import BaseSkill`` itself. Everything else is
blocked by AST validation and by the child process import gate.

Every code path through run() must end with ``canvas.commit(image)``.
"""

from __future__ import annotations


class BaseSkill:
    """The contract every skill implements.

    Class attributes (override these):
        name:
            User-facing title. The slug (lowercased, underscore-separated)
            is derived from this.
        description:
            Short searchable description shown in the catalog and used for
            semantic search.
        kind:
            Either "creation" (produces a new image from scratch) or
            "transform" (takes the current canvas and reshapes it).
        owner:
            Session key of the author. Empty for built-ins, or set to
            "library" by convention.
        created_at:
            Epoch seconds. Set automatically when written via the
            create_skill tool.
        controls:
            Optional list of user-facing controls (slider/enum/bool/pan/
            palette). Validated against the run() signature. Max 3
            non-palette controls; a palette swatch is auto-added when the
            skill references canvas.palette or art_kit.palette_color.
        hidden:
            Soft-delete flag. Hidden skills still load (so shared canvas
            chains can replay) but are excluded from list/search.

    Methods (override these):
        run(canvas, **params)
            The skill body. Must call canvas.commit(image) before returning.
    """

    # --- Identity ---
    name: str = ""
    description: str = ""
    kind: str = "creation"          # "creation" | "transform"
    owner: str = ""
    created_at: float = 0.0

    # --- UI / catalog ---
    controls: list = []
    hidden: bool = False

    # --- Discovery parity with other plugin base classes ---
    auto_register: bool = True
    requires_services: list[str] = []
    config_settings: list = []

    def __init_subclass__(cls, **kwargs):
        """Defensive copies so subclasses don't mutate base-class containers."""
        super().__init_subclass__(**kwargs)
        for attr in ("controls", "requires_services", "config_settings"):
            value = getattr(cls, attr)
            if isinstance(value, (list, dict)):
                setattr(cls, attr, value.copy())

    @property
    def slug(self) -> str:
        """Slugified form of `name`, used as the catalog key."""
        from plugins.skills.helpers.skill_store import slugify
        return slugify(self.name)

    def run(self, canvas, **params):
        """Execute the skill. Must call canvas.commit(image)."""
        raise NotImplementedError(f"Skill '{self.name}' must implement run()")
