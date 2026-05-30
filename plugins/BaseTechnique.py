"""Base class for canvas techniques.

A technique is a Python file under plugins/techniques/ (baked-in) or
DATA_DIR/sandbox_techniques/ (agent/user authored) that defines a class
subclassing BaseTechnique. The class declares metadata as class attributes and
implements `run(canvas)` to either create or transform an image.

Technique code is executed in a subprocess sandbox (plugins/techniques/helpers/
technique_sandbox_entry.py) with restricted imports and resource limits. The
sandbox imports the file, finds the BaseTechnique subclass, instantiates it,
sets descriptor-declared controls on the instance, and calls run(canvas).

Allowed imports inside a technique: math, random, colorsys, numpy, PIL.*, and
``from plugins.BaseTechnique import BaseTechnique, Slider, Bool, Enum, Pan, Palette, Text``.
Everything else is
blocked by AST validation and by the child process import gate.

Every code path through run() must end with ``canvas.commit(image)``.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Control descriptors.
#
# These let library techniques declare controls as class attributes:
#
#     class Fisheye(BaseTechnique):
#         strength = Slider(-1.0, 1.0, default=0.6)
#         mode     = Enum(['radial', 'uniform'], default='radial')
#
#         def run(self, canvas):
#             # self.strength / self.mode already populated + clamped
#             ...
#
# Descriptors are pure metadata holders — no __get__/__set__. ``BaseTechnique.
# __init_subclass__`` scans for them and compiles the runtime UI schema.
# The sandbox dispatcher sets the corresponding attributes on the instance
# before calling run(), so ``self.x`` reads the clamped current value.
# ---------------------------------------------------------------------------


class _ControlDescriptor:
    """Metadata-only base for control descriptors."""
    control_type: str = ""


class Slider(_ControlDescriptor):
    """Numeric slider control. ``min`` and ``max`` define both the UI range
    and the auto-clamp bounds applied to ``self.<name>`` at dispatch time.

    Example:
        strength = Slider(-1.0, 1.0, default=0.6)
        radius   = Slider(1, 80, default=18, step=1)
    """
    control_type = "slider"

    def __init__(self, min, max, default=None, *, step=None, label=None):
        if max <= min:
            raise ValueError(f"Slider needs max > min (got min={min}, max={max})")
        self.min = float(min)
        self.max = float(max)
        self.default = float(default if default is not None else min)
        self.step = float(step if step is not None else (max - min) / 100.0)
        self.label = label


class Bool(_ControlDescriptor):
    """Boolean toggle control."""
    control_type = "bool"

    def __init__(self, default=False, *, label=None):
        self.default = bool(default)
        self.label = label


class Enum(_ControlDescriptor):
    """Multi-choice dropdown control.

    ``options`` may be a list of strings (label == value) or a list of
    ``(value, label)`` 2-tuples / dicts.
    """
    control_type = "enum"

    def __init__(self, options, default=None, *, label=None):
        if not options:
            raise ValueError("Enum needs at least one option")
        normalized = []
        for o in options:
            if isinstance(o, str):
                normalized.append((o, o))
            elif isinstance(o, dict):
                normalized.append((o["value"], str(o.get("label", o["value"]))))
            else:
                value, lbl = o[0], (o[1] if len(o) > 1 else o[0])
                normalized.append((value, str(lbl)))
        self.options = normalized
        self.default = default if default is not None else normalized[0][0]
        self.label = label


class Pan(_ControlDescriptor):
    """Two-axis arrow-pad widget that drives two underlying Sliders.

    ``x`` and ``y`` are names of *other* attributes on the same class —
    both must be Sliders. Pan does NOT introduce its own injected
    attribute; ``self.<pan_name>`` is unset. Read the underlying scalars.

    Example:
        cx     = Slider(0, 1, default=0.5)
        cy     = Slider(0, 1, default=0.5)
        center = Pan(x='cx', y='cy')
    """
    control_type = "pan"

    def __init__(self, x, y, *, label=None, step=None):
        self.x_param = str(x)
        self.y_param = str(y)
        self.label = label
        self.step = step  # None → inherit from the underlying slider step


class Text(_ControlDescriptor):
    """Single-line free-text input. Clamped to ``max_length`` characters."""
    control_type = "text"

    def __init__(self, default="", *, max_length=120, placeholder=None, label=None):
        self.default = str(default)
        self.max_length = int(max_length)
        self.placeholder = placeholder
        self.label = label


class Palette(_ControlDescriptor):
    """Per-layer palette override control. Declares that the technique should
    expose a palette swatch. No injected attribute; the runtime resolves
    the palette before run() is called.
    """
    control_type = "palette"

    def __init__(self, *, label=None):
        self.label = label


_RESERVED_ATTRS = frozenset({
    "name", "description", "kind", "owner", "created_at",
    "controls", "hidden", "auto_register", "requires_services",
    "config_settings", "slug", "run", "_param_bounds",
})


def _compile_descriptors(cls) -> tuple[list[dict], dict[str, dict]] | None:
    """Walk ``cls.__dict__`` for ``_ControlDescriptor`` instances and return
    ``(controls, param_bounds)``:

      * ``controls`` is the runtime UI representation. Sliders consumed by
        a Pan as ``x_param``/``y_param`` are not emitted as separate UI
        entries — the Pan widget drives them.
      * ``param_bounds`` is a dispatch-time table keyed by attribute name,
        carrying ``{type, min, max, default}`` (or ``{type, default}`` for
        bool/enum) for *every* declared descriptor. The sandbox dispatcher
        uses this to clamp + default each ``self.<name>`` before run().

    Returns None when no descriptors are declared.
    """
    descriptors: list[tuple[str, _ControlDescriptor]] = []
    sliders: dict[str, Slider] = {}
    pans: list[tuple[str, Pan]] = []
    for attr_name, value in cls.__dict__.items():
        if isinstance(value, _ControlDescriptor):
            if attr_name in _RESERVED_ATTRS:
                raise TypeError(
                    f"{cls.__name__}: control name '{attr_name}' collides with "
                    f"a BaseTechnique metadata attribute; pick another name "
                    f"(e.g. '{attr_name}_kind' or a semantic synonym)"
                )
            descriptors.append((attr_name, value))
            if isinstance(value, Slider):
                sliders[attr_name] = value
            if isinstance(value, Pan):
                pans.append((attr_name, value))
    if not descriptors:
        return None

    # Names of sliders absorbed by a Pan — they don't get their own UI entry.
    pan_consumed: set[str] = set()
    for _, pan in pans:
        x = sliders.get(pan.x_param)
        y = sliders.get(pan.y_param)
        if x is None or y is None:
            raise TypeError(
                f"Pan on {cls.__name__} references x={pan.x_param!r}/"
                f"y={pan.y_param!r}; both must be Slider attributes on "
                f"the same class"
            )
        pan_consumed.add(pan.x_param)
        pan_consumed.add(pan.y_param)

    controls: list[dict] = []
    param_bounds: dict[str, dict] = {}
    for attr_name, d in descriptors:
        label = d.label or attr_name.replace("_", " ").title()
        if isinstance(d, Slider):
            param_bounds[attr_name] = {
                "type": "slider", "min": d.min, "max": d.max,
                "default": d.default,
            }
            if attr_name in pan_consumed:
                continue
            controls.append({
                "type": "slider",
                "name": attr_name,
                "label": label,
                "min": d.min,
                "max": d.max,
                "step": d.step,
                "default": d.default,
            })
        elif isinstance(d, Bool):
            param_bounds[attr_name] = {"type": "bool", "default": d.default}
            controls.append({
                "type": "bool",
                "name": attr_name,
                "label": label,
                "default": d.default,
            })
        elif isinstance(d, Enum):
            allowed = [v for (v, _) in d.options]
            param_bounds[attr_name] = {
                "type": "enum", "default": d.default, "allowed": allowed,
            }
            controls.append({
                "type": "enum",
                "name": attr_name,
                "label": label,
                "options": [{"value": v, "label": l} for (v, l) in d.options],
                "default": d.default,
            })
        elif isinstance(d, Pan):
            x_slider = sliders[d.x_param]
            y_slider = sliders[d.y_param]
            step = d.step if d.step is not None else min(x_slider.step, y_slider.step)
            controls.append({
                "type": "pan",
                "name": attr_name,
                "label": label,
                "x_param": d.x_param,
                "y_param": d.y_param,
                "step": float(step),
                "x_default": x_slider.default,
                "y_default": y_slider.default,
            })
        elif isinstance(d, Text):
            param_bounds[attr_name] = {
                "type": "text", "default": d.default, "max_length": d.max_length,
            }
            controls.append({
                "type": "text",
                "name": attr_name,
                "label": label,
                "default": d.default,
                "max_length": d.max_length,
                "placeholder": d.placeholder,
            })
        elif isinstance(d, Palette):
            controls.append({
                "type": "palette",
                "name": "palette",
                "label": label,
            })
    return controls, param_bounds


class BaseTechnique:
    """The contract every technique implements.

    Class attributes (override these):
        name:
            User-facing title. The slug (lowercased, underscore-separated)
            is derived from this.
        description:
            Short searchable description shown in the catalog and used for
            semantic search.
        kind:
            One of "background" (produces a new image from scratch),
            "filter" (takes the current canvas and reshapes it, returning
            a same-shape opaque image that replaces it), or "object" (reads
            the current canvas, returns an RGBA image that the framework
            alpha-composites onto the prior canvas — used for overlays like
            typography). Objects and filters both require a prior layer;
            only backgrounds may sit at layer 0.
        owner:
            Session key of the author. Defaults to "library" for techniques
            shipped in plugins/techniques/; sandbox-authored techniques set it to
            the author's session key via ``write_technique``.
        created_at:
            Epoch seconds. Set automatically when written via the
            create_technique tool. ``0.0`` triggers a fallback to file mtime.
        controls:
            Optional list of user-facing controls (slider/enum/bool/pan/
            palette). Max 4
            non-palette controls; add a palette control only when the
            technique uses palette and should expose a layer override.

            Declare each control as a class-attribute descriptor —
            ``strength = Slider(-1.0, 1.0, default=0.6)`` — and the
            framework compiles this list automatically. Read the values
            inside ``run()`` as ``self.strength``.
            Auto-clamping to the declared range is applied at dispatch.
        hidden:
            Soft-delete flag. Hidden techniques still load (so shared canvas
            chains can replay) but are excluded from list/search.

    Methods (override these):
        run(canvas, **params)
            The technique body. Must call canvas.commit(image) before returning.
    """

    # --- Identity ---
    name: str = ""
    description: str = ""
    kind: str = "background"        # "background" | "filter" | "object"
    owner: str = "library"          # library techniques can omit this entirely
    created_at: float = 0.0

    # --- UI / catalog ---
    controls: list = []
    hidden: bool = False

    # --- Discovery parity with other plugin base classes ---
    auto_register: bool = True
    requires_services: list[str] = []
    config_settings: list = []

    # Dispatcher-facing bounds table for descriptor-declared parameters.
    # Populated by ``__init_subclass__``; consumed by the sandbox to clamp
    # + default each ``self.<name>`` before ``run()`` is called.
    _param_bounds: dict = {}

    def __init_subclass__(cls, **kwargs):
        """Defensive copies so subclasses don't mutate base-class containers;
        compile control descriptors into the runtime ``controls`` list."""
        super().__init_subclass__(**kwargs)
        if "controls" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}: declare controls with Slider/Enum/Bool/Pan/"
                "Text/Palette descriptors, not a literal controls list"
            )
        run = cls.__dict__.get("run")
        code = getattr(run, "__code__", None)
        if code and (
            code.co_argcount != 2
            or code.co_varnames[:2] != ("self", "canvas")
            or code.co_kwonlyargcount
            or code.co_flags & 0x0C
        ):
            raise TypeError(f"{cls.__name__}: run must be declared exactly as def run(self, canvas)")

        for attr in ("requires_services", "config_settings"):
            value = getattr(cls, attr)
            if isinstance(value, (list, dict)):
                setattr(cls, attr, value.copy())

        compiled = _compile_descriptors(cls)
        if compiled is not None:
            controls, param_bounds = compiled
            cls.controls = controls
            cls._param_bounds = param_bounds
        else:
            cls.controls = []
            cls._param_bounds = {}

    @property
    def slug(self) -> str:
        """Slugified form of `name`, used as the catalog key."""
        from plugins.techniques.helpers.technique_store import slugify
        return slugify(self.name)

    def run(self, canvas):
        """Execute the technique. Must call canvas.commit(image)."""
        raise NotImplementedError(f"Technique '{self.name}' must implement run()")
