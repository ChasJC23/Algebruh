from abc import ABC, abstractmethod
import cmath
from typing import Callable, Iterable, TypeAlias
from . import parser
from . import nodes

Expression: TypeAlias = "nodes.Node"
'''
Some arbitrary mathematical expression
'''

class Context(ABC):
    '''
    class encapsulating available functions and variables with known value.
    '''
    @abstractmethod
    def resolveVariable(self, variable: str) -> complex:
        '''
        resolves the value of a known identifier within the expression.
        '''

    def resolveFunction(self, function: str) -> Callable[..., complex]:
        '''
        resolves the operation of a known identifier within the expression.
        '''
        if function == "ln":
            return cmath.log
    @abstractmethod
    def differentiateFtn(self, function: str) -> Iterable[nodes.PartialExpression]:
        '''
        returns the derivative function(s) of the given function.
        '''

class StdContext(Context):
    '''
    standard mathematical context.
    '''
    def __init__(self, **variables: complex):
        self.variables = variables

    def resolveVariable(self, variable: str) -> complex:
        if variable in self.variables.keys():
            return self.variables[variable]
        else:
            raise SyntaxError()

    def resolveFunction(self, function: str) -> Callable[..., complex]:
        try:
            resolved = getattr(cmath, function)
        except:
            resolved = None
        if callable(resolved):
            return resolved
        match function:
            case "ln": return cmath.log
            case "sec": return lambda x: 1 / cmath.cos(x)
            case "csc": return lambda x: 1 / cmath.sin(x)
            case "cot": return lambda x: 1 / cmath.tan(x)
            case "sech": return lambda x: 1 / cmath.cosh(x)
            case "csch": return lambda x: 1 / cmath.sinh(x)
            case "coth": return lambda x: 1 / cmath.tanh(x)
        raise SyntaxError()

    def differentiateFtn(self, function: str) -> Iterable[nodes.PartialExpression]:
        match function:
            case "sqrt": return (nodes.PartialExpression(1 / (2 * nodes.FunctionCallNode("sqrt", nodes.PlaceholderNode(0)))),)
            case "rect": return (nodes.PartialExpression(nodes.FunctionCallNode("exp", nodes.PlaceholderNode(1))), nodes.PartialExpression(nodes.FunctionCallNode("rect", nodes.PlaceholderNode(0), nodes.PlaceholderNode(1))))
            case "exp": return (nodes.PartialExpression(nodes.FunctionCallNode("exp", nodes.PlaceholderNode(0))),)
            case "ln" | "log": return (nodes.PartialExpression(1 / nodes.PlaceholderNode(0)),)
            case "log10": return (nodes.PartialExpression(1 / (nodes.PlaceholderNode(0) * nodes.FunctionCallNode("log", 10))),)
            case "sin": return (nodes.PartialExpression(nodes.FunctionCallNode("cos", nodes.PlaceholderNode(0))),)
            case "cos": return (nodes.PartialExpression(-nodes.FunctionCallNode("sin", nodes.PlaceholderNode(0))),)
            case "tan": return (nodes.PartialExpression(nodes.FunctionCallNode("sec", nodes.PlaceholderNode(0)) ** 2),)
            case "sec": return (nodes.PartialExpression(nodes.FunctionCallNode("tan", nodes.PlaceholderNode(0)) * nodes.FunctionCallNode("sec", nodes.PlaceholderNode(0))),)
            case "csc": return (nodes.PartialExpression(-(nodes.FunctionCallNode("cot", nodes.PlaceholderNode(0)) * nodes.FunctionCallNode("csc", nodes.PlaceholderNode(0)))),)
            case "cot": return (nodes.PartialExpression(-nodes.FunctionCallNode("csc", nodes.PlaceholderNode(0)) ** 2),)
            case "asin": return (nodes.PartialExpression(1 / nodes.FunctionCallNode("sqrt", 1 - nodes.PlaceholderNode(0) ** 2)),)
            case "acos": return (nodes.PartialExpression(-1 / nodes.FunctionCallNode("sqrt", 1 - nodes.PlaceholderNode(0) ** 2)),)
            case "atan": return (nodes.PartialExpression(1 / (1 + nodes.PlaceholderNode(0) ** 2)),)
            case "sinh": return (nodes.PartialExpression(nodes.FunctionCallNode("cosh", nodes.PlaceholderNode(0))))
            case "cosh": return (nodes.PartialExpression(nodes.FunctionCallNode("sinh", nodes.PlaceholderNode(0))))
            case "tanh": return (nodes.PartialExpression(nodes.FunctionCallNode("sech", nodes.PlaceholderNode(0)) ** 2),)
            case "sech": return (nodes.PartialExpression(-(nodes.FunctionCallNode("tanh", nodes.PlaceholderNode(0)) * nodes.FunctionCallNode("sech", nodes.PlaceholderNode(0)))),)
            case "csch": return (nodes.PartialExpression(-(nodes.FunctionCallNode("coth", nodes.PlaceholderNode(0)) * nodes.FunctionCallNode("csch", nodes.PlaceholderNode(0)))),)
            case "coth": return (nodes.PartialExpression(-nodes.FunctionCallNode("csch", nodes.PlaceholderNode(0)) ** 2),)
            case "asinh": return (nodes.PartialExpression(1 / nodes.FunctionCallNode("sqrt", nodes.PlaceholderNode(0) ** 2 + 1)),)
            case "acosh": return (nodes.PartialExpression(1 / nodes.FunctionCallNode("sqrt", (nodes.PlaceholderNode(0) ** 2 - 1) * (nodes.PlaceholderNode(0) ** 2 + 1))),)
            case "atanh": return (nodes.PartialExpression(1 / (1 - nodes.PlaceholderNode(0) ** 2)),)

def expression(expr: str) -> Expression:
    return parser.Parser.parse(expr)