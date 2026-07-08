import pprint
from typing import Literal

from hoid.core.tools import build_schema, tool

# --- build_schema type coverage ---


def _schema_for(fn):
    return build_schema(fn)["function"]["parameters"]["properties"]


def _required_for(fn):
    return build_schema(fn)["function"]["parameters"]["required"]


def test_str_param():
    def f(name: str): ...

    prop = _schema_for(f)["name"]
    print(f"str param schema: {prop}")
    assert prop == {"type": "string"}


def test_int_param():
    def f(n: int): ...

    prop = _schema_for(f)["n"]
    print(f"int param schema: {prop}")
    assert prop == {"type": "integer"}


def test_float_param():
    def f(x: float): ...

    prop = _schema_for(f)["x"]
    print(f"float param schema: {prop}")
    assert prop == {"type": "number"}


def test_bool_param():
    def f(flag: bool): ...

    prop = _schema_for(f)["flag"]
    print(f"bool param schema: {prop}")
    assert prop == {"type": "boolean"}


def test_list_of_str():
    def f(items: list[str]): ...

    prop = _schema_for(f)["items"]
    print(f"list[str] param schema: {prop}")
    assert prop == {"type": "array", "items": {"type": "string"}}


def test_plain_list():
    def f(items: list): ...

    assert _schema_for(f)["items"] == {"type": "array"}


def test_dict_param():
    def f(data: dict): ...

    assert _schema_for(f)["data"] == {"type": "object"}


def test_optional_unwrapped():
    def f(value: str | None): ...

    prop = _schema_for(f)["value"]
    print(f"Optional[str] param schema: {prop}")
    assert prop == {"type": "string"}


def test_literal_becomes_enum():
    def f(color: Literal["red", "green", "blue"]): ...

    schema = _schema_for(f)["color"]
    print(f"Literal param schema: {schema}")
    assert schema["type"] == "string"
    assert set(schema["enum"]) == {"red", "green", "blue"}


def test_param_with_default_not_required():
    def f(required: str, optional: str = "default"): ...

    req = _required_for(f)
    print(f"required fields: {req}")
    assert "required" in req
    assert "optional" not in req


def test_context_param_excluded():
    def f(task: str, _context: object): ...

    props = _schema_for(f)
    print(f"props keys (should not have '_context'): {list(props.keys())}")
    assert "_context" not in props
    assert "task" in props


# --- @tool decorator ---


def test_tool_attaches_schema():
    @tool
    def my_func(x: int) -> str:
        "Does something."
        return str(x)

    print("my_func.schema:")
    pprint.pprint(my_func.schema)
    assert hasattr(my_func, "schema")
    assert my_func.schema["function"]["name"] == "my_func"


def test_tool_attaches_name_and_description():
    @tool
    def greet(name: str) -> str:
        "Greet someone by name."
        return f"Hello {name}"

    print(f"greet.name={greet.name!r}  greet.description={greet.description!r}")
    assert greet.name == "greet"
    assert greet.description == "Greet someone by name."


def test_tool_callable_after_decoration():
    @tool
    def add(a: int, b: int) -> int:
        "Add two integers."
        return a + b

    assert add(2, 3) == 5


# --- Args: docstring parsing ---


def test_args_block_populates_descriptions():
    @tool
    def upload(path: str, overwrite: bool = False) -> str:
        """Upload a file.

        Args:
            path: The file path to upload.
            overwrite: Replace existing files if True.
        """
        return "ok"

    props = upload.schema["function"]["parameters"]["properties"]
    print(f"path description: {props['path'].get('description')!r}")
    print(f"overwrite description: {props['overwrite'].get('description')!r}")
    assert props["path"].get("description") == "The file path to upload."
    assert props["overwrite"].get("description") == "Replace existing files if True."


def test_no_docstring_uses_empty_description():
    def f(x: str):
        pass

    schema = build_schema(f)
    desc = schema["function"]["description"]
    print(f"description for no-docstring fn: {desc!r}")
    assert desc == ""


def test_function_description_is_first_line():
    @tool
    def compute(x: float) -> float:
        """Compute the result.

        Args:
            x: Input value.
        """
        return x * 2

    assert compute.schema["function"]["description"] == "Compute the result."
