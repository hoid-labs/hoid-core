import functools
import inspect
import typing
from typing import get_type_hints

# maps Python built-in types to their JSON Schema type strings
_PY_TO_JSON = {
    int: "integer",
    float: "number",
    str: "string",
    bool: "boolean",
    list: "array",
    dict: "object",
}


# --- type resolution ---


def _type_to_schema(t):
    origin = typing.get_origin(t)

    if origin is typing.Union:
        # Optional[X] is Union[X, None] — unwrap to just X
        args = [a for a in typing.get_args(t) if a is not type(None)]
        return _type_to_schema(args[0]) if len(args) == 1 else {"type": "string"}

    if origin is list:
        args = typing.get_args(t)
        return (
            {"type": "array", "items": _type_to_schema(args[0])}
            if args
            else {"type": "array"}
        )

    if origin is dict:
        return {"type": "object"}

    if origin is typing.Literal:
        # Literal[a, b] maps to a string enum
        return {"type": "string", "enum": list(typing.get_args(t))}

    return {"type": _PY_TO_JSON.get(t, "string")}


# --- schema builder ---


def build_schema(fn):
    "Build an OpenAI-compatible JSON function schema from a function's type hints and docstring."
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except NameError as e:
        raise TypeError(f"Unresolvable annotation in '{fn.__name__}': {e}") from e

    doc = inspect.cleandoc(fn.__doc__ or "")
    lines = doc.splitlines()
    desc = lines[0] if lines else ""

    param_descs, in_args = {}, False
    # parse the Args: block for per-parameter descriptions
    for line in lines[1:]:
        s = line.strip()
        if s.lower() in ("args:", "arguments:"):
            in_args = True
        elif in_args and ":" in s:
            name, _, rest = s.partition(":")
            param_descs[name.strip()] = rest.strip()

    props, required = {}, []
    for name, param in sig.parameters.items():
        # injected by FastMCP at runtime, not part of the schema
        if name == "_context":
            continue

        # no JSON Schema equivalent for variadic params
        if param.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            continue

        prop = _type_to_schema(hints.get(name, str))
        if name in param_descs:
            prop["description"] = param_descs[name]
        props[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn.__name__,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


# --- decorators ---


def tool(fn):
    "Decorator that attaches a JSON schema to a function, making it usable as an agent tool."
    fn.schema = build_schema(fn)
    fn.description = fn.schema["function"]["description"]
    fn.name = fn.schema["function"]["name"]
    return fn


def _make_key(value):
    # recursively convert unhashable containers so the result can be used as a dict key
    if isinstance(value, list):
        return tuple(_make_key(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _make_key(v)) for k, v in value.items()))
    return value


def cached_tool(fn=None, *, maxsize=128):
    "Decorator that caches tool results for the lifetime of the process; safe for both sync and async tools."

    def decorator(fn):
        schema = build_schema(fn)
        _cache: dict = {}

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def wrapper(*args, **kwargs):
                # sorted kwargs makes the key order-independent
                key = (_make_key(args), _make_key(tuple(sorted(kwargs.items()))))
                if key in _cache:
                    # move to end to record recent use for LRU eviction
                    _cache[key] = _cache.pop(key)
                else:
                    if maxsize is not None and len(_cache) >= maxsize:
                        # evict least recently used entry (dict preserves insertion order)
                        _cache.pop(next(iter(_cache)))
                    _cache[key] = await fn(*args, **kwargs)
                return _cache[key]

        else:

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                # sorted kwargs makes the key order-independent
                key = (_make_key(args), _make_key(tuple(sorted(kwargs.items()))))
                if key in _cache:
                    # move to end to record recent use for LRU eviction
                    _cache[key] = _cache.pop(key)
                else:
                    if maxsize is not None and len(_cache) >= maxsize:
                        _cache.pop(next(iter(_cache)))
                    _cache[key] = fn(*args, **kwargs)
                return _cache[key]

        wrapper.schema = schema
        wrapper.description = schema["function"]["description"]
        wrapper.name = schema["function"]["name"]
        return wrapper

    # supports both @cached_tool and @cached_tool(maxsize=256)
    if fn is not None:
        return decorator(fn)
    return decorator
