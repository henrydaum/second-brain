"""Browse and run workspace scripts without a model round trip.

Inspect source with AST, never import it to build a form. Scripts still run
through script.run and its normal validation and permission gates.
Output: scripts may call sdk.ui.render themselves, or return a dict with
an explicit `attachments` list and optional `text`. Other values are printed.
"""
import ast
import json
from guest.bases import BaseCommand
from guest.forms import FormStep


def describe(source):
    """Read main's keyword-callable parameters without executing any code."""
    tree = ast.parse(source)
    main = next((n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "main"), None)
    if main is None:
        raise ValueError("No main(sdk, ...) function found.")
    if isinstance(main, ast.AsyncFunctionDef):
        raise ValueError("Async main functions are not supported by this command.")
    positional = list(main.args.posonlyargs) + list(main.args.args)
    if not positional or positional[0].arg != "sdk":
        raise ValueError("The first parameter must be sdk.")
    if len(main.args.posonlyargs) > 1:
        raise ValueError("Script inputs must accept keyword arguments.")
    if main.args.vararg or main.args.kwarg:
        raise ValueError("Use named parameters instead of *args or **kwargs for interactive scripts.")
    defaults = [None] * (len(positional) - len(main.args.defaults)) + list(main.args.defaults)
    pairs = list(zip(positional[1:], defaults[1:])) + list(zip(main.args.kwonlyargs, main.args.kw_defaults))
    fields = []
    for param, default in pairs:
        if param.arg in ("path", "entry", "wait"):
            raise ValueError(f"Parameter {param.arg!r} conflicts with the script runner; rename it for interactive use.")
        required = default is None
        value = None
        if default is not None:
            try:
                value = ast.literal_eval(default)
            except (ValueError, TypeError):
                pass  # Blank omits the argument: Python evaluates its own default.
        annotation = param.annotation.id if isinstance(param.annotation, ast.Name) else ""
        kind = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "dict": "object"}.get(annotation)
        if kind is None:
            kind = "boolean" if isinstance(value, bool) else "integer" if isinstance(value, int) else "number" if isinstance(value, float) else "array" if isinstance(value, list) else "object" if isinstance(value, dict) else "string"
        fields.append({"parameter": param.arg, "name": "arg_" + param.arg,
                       "type": kind, "required": required, "default": value})
    return ast.get_docstring(main) or ast.get_docstring(tree) or "No description.", fields


def scripts(sdk):
    root = sdk.paths.get("scripts")
    return {sdk.path.name(p): p for p in sdk.fs.list(root, pattern="*.py")}


class ScriptsCommand(BaseCommand):
    name = "scripts"
    description = "Select a script, inspect its inputs, and run it"
    category = "Capabilities"
    requests = ["paths.get", "fs.list", "fs.read", "script.run", "ui.render", "ui.progress", "fs.delete"]
    approval_actions = ("delete",)
    timeout = 600

    def form(self, sdk, args):
        available = scripts(sdk)
        if not available:
            return []
        steps = [FormStep("script_name", "Select a script.", True,
                          enum=sorted(available), columns=2)]
        path = available.get(args.get("script_name"))
        if path is None:
            return steps
        try:
            description, fields = describe(sdk.fs.read(path))
        except (SyntaxError, ValueError) as exc:
            return steps + [FormStep("action", str(exc), True, enum=["inspect"], enum_labels=["Show details"])]
        steps.append(FormStep("action", "What do you want to do with this script?\n\n" + description,
                              True, enum=["run", "delete"], enum_labels=["Run script", "Delete script"]))
        if args.get("action") == "run":
            for field in fields:
                name = field["parameter"]
                kind = field["type"]
                prompt = f"Enter {name.replace('_', ' ')}."
                if kind in ("array", "object"):
                    prompt += " Use a JSON " + kind + "."
                if not field["required"]:
                    prompt += " Leave blank to use the script's default."
                    if field["default"] is not None:
                        prompt += " Default: " + json.dumps(field["default"], default=str)
                steps.append(FormStep(field["name"], prompt,
                                      field["required"],
                                      type="string" if kind in ("array", "object") else kind,
                                      prompt_when_missing=True))
        return steps

    def run(self, sdk, args):
        available = scripts(sdk)
        if not available:
            return "No scripts found in " + sdk.paths.get("scripts") + "."
        path = available.get(args.get("script_name"))
        if path is None:
            return "Unknown script. Run /scripts to choose one."
        try:
            description, fields = describe(sdk.fs.read(path))
        except (SyntaxError, ValueError) as exc:
            return str(exc)
        if args.get("action") == "delete":
            sdk.fs.delete(path)
            return "Deleted " + args["script_name"] + "."
        if args.get("action") != "run":
            return description
        values = {}
        for field in fields:
            value = args.get(field["name"])
            if value is None or value == "":
                if field["required"]:
                    return "Missing argument: " + field["parameter"]
                continue
            if field["type"] in ("array", "object"):
                try:
                    value = json.loads(value) if isinstance(value, str) else value
                except ValueError:
                    return "Invalid JSON for " + field["parameter"]
                expected = list if field["type"] == "array" else dict
                if not isinstance(value, expected):
                    return "Expected a JSON " + field["type"] + " for " + field["parameter"]
            values[field["parameter"]] = value
        sdk.ui.progress("Running " + args["script_name"] + "…")
        result = sdk.scripts.run(path, **values)
        if isinstance(result, dict) and isinstance(result.get("attachments"), list):
            attachments = result["attachments"]
            if not all(isinstance(p, str) for p in attachments):
                raise ValueError("Returned attachments must be a list of file paths.")
            if attachments:
                sdk.ui.render(attachments)
            result = {k: v for k, v in result.items() if k != "attachments"}
            if set(result) == {"text"}:
                return str(result["text"])
        if result is None or result == {}:
            return "Finished " + args["script_name"] + "."
        if isinstance(result, str):
            return result
        return "```json\n" + json.dumps(result, indent=2, default=str) + "\n```"
