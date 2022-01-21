from __future__ import annotations
from typing import TypeAlias, TYPE_CHECKING
from . import parser
if TYPE_CHECKING:
    from . import nodes

Expression: TypeAlias = nodes.Node
'''
Some arbitrary mathematical expression
'''

def expression(expr: str) -> Expression:
    return parser.Parser.parse(expr)