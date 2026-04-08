"""
Internal infrastructure for safely cancelling long-running state execution
loops by injecting shutdown checks into user-defined code.

This system automatically modifies the AST (Abstract Syntax Tree) of state
`execute()` methods at class creation time, inserting cancellation checks
inside all `for` and `while` loops.
"""

from abc import ABCMeta
import ast
import inspect
import textwrap
from typing import Any, Dict


class LoopCancellationInjector(ast.NodeTransformer):
    """
    AST transformer that injects shutdown checks into loop bodies.

    This transformer modifies:
    - ``while`` loops
    - ``for`` loops

    by inserting a call to ``self.check_shutdown()`` as the first statement
    in each loop body.
    """
    def create_check_node(self, lineno=0, col_offset=0) -> ast.Expr:
        return ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="self", ctx=ast.Load()),
                    attr="check_shutdown",
                    ctx=ast.Load()
                ),
                args=[],
                keywords=[]
            ),
            lineno=lineno,
            col_offset=col_offset
        )

    def visit_While(self, node) -> Any:
        self.generic_visit(node)
        check_node = self.create_check_node(
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0)
        )
        node.body.insert(0, check_node)
        return node
    
    def visit_For(self, node) -> Any:
        self.generic_visit(node)
        check_node = self.create_check_node(
            lineno=getattr(node, "lineno", 0),
            col_offset=getattr(node, "col_offset", 0)
        )
        node.body.insert(0, check_node)
        return node


class CancellableMeta(ABCMeta):
    """
    Metaclass that injects loop cancellation logic into `execute()` methods.

    If a class defines an ``execute`` method:
    - Its source code is retrieved
    - Parsed into an AST
    - All loops are instrumented with shutdown checks
    - The modified function replaces the original
    """
    def __new__(mcls, name, bases, attrs):
        if "execute" in attrs and inspect.isfunction(attrs["execute"]):
            original = attrs["execute"]

            try:
                src = inspect.getsource(original)
                src = textwrap.dedent(src)
                tree = ast.parse(src)

                injector = LoopCancellationInjector()
                new_tree = injector.visit(tree)
                ast.fix_missing_locations(new_tree)

                env = original.__globals__.copy()
                local_env: Dict[str, Any] = {}

                compiled = compile(new_tree, filename="<ast>", mode="exec")
                exec(compiled, env, local_env)

                new_func = local_env.get(original.__name__)
                if new_func is not None:
                    new_func.__defaults__ = original.__defaults__
                    new_func.__kwdefaults__ = original.__kwdefaults__
                    attrs["execute"] = new_func
                else:
                    attrs["execute"] = original

            except (OSError, IOError, TypeError, IndentationError, SyntaxError, ValueError):
                attrs["execute"] = original

        return super().__new__(mcls, name, bases, attrs)
