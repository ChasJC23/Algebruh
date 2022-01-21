from abc import ABC, abstractmethod
import cmath
from typing import TypeAlias, Callable, Iterable
from . import parser, nodes

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
    def differentiateFtn(self, function: str, *args: nodes.Node) -> Iterable[nodes.Node]:
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

    def differentiateFtn(self, function: str, *args: nodes.Node) -> Iterable[nodes.Node]:
        match function:
            case "sqrt": return (1 / (2 * nodes.FunctionCallNode("sqrt", args[0])),)
            case "rect": return (nodes.FunctionCallNode("exp", args[1]), nodes.FunctionCallNode("rect", args[0], args[1]))
            case "exp": return (nodes.FunctionCallNode("exp", args[0]),)
            case "ln" | "log": return (1 / args[0],)
            case "log10": return (1 / (args[0] * nodes.FunctionCallNode("log", 10)),)
            case "sin": return (nodes.FunctionCallNode("cos", args[0]),)
            case "cos": return (-nodes.FunctionCallNode("sin", args[0]),)
            case "tan": return (nodes.FunctionCallNode("sec", args[0]) ** 2,)
            case "sec": return (nodes.FunctionCallNode("tan", args[0]) * nodes.FunctionCallNode("sec", args[0]),)
            case "csc": return (-(nodes.FunctionCallNode("cot", args[0]) * nodes.FunctionCallNode("csc", args[0])),)
            case "cot": return (-nodes.FunctionCallNode("csc", args[0]) ** 2,)
            case "asin": return (1 / nodes.FunctionCallNode("sqrt", 1 - args[0] ** 2),)
            case "acos": return (-1 / nodes.FunctionCallNode("sqrt", 1 - args[0] ** 2),)
            case "atan": return (1 / (1 + args[0] ** 2),)
            case "sinh": return (nodes.FunctionCallNode("cosh", args[0]))
            case "cosh": return (nodes.FunctionCallNode("sinh", args[0]))
            case "tanh": return (nodes.FunctionCallNode("sech", args[0]) ** 2,)
            case "sech": return (-(nodes.FunctionCallNode("tanh", args[0]) * nodes.FunctionCallNode("sech", args[0])),)
            case "csch": return (-(nodes.FunctionCallNode("coth", args[0]) * nodes.FunctionCallNode("csch", args[0])),)
            case "coth": return (-nodes.FunctionCallNode("csch", args[0]) ** 2,)
            case "asinh": return (1 / nodes.FunctionCallNode("sqrt", args[0] ** 2 + 1),)
            case "acosh": return (1 / nodes.FunctionCallNode("sqrt", (args[0] ** 2 - 1) * (args[0] ** 2 + 1)),)
            case "atanh": return (1 / (1 - args[0] ** 2),)

def expression(expr: str) -> Expression:
    return parser.Parser.parse(expr)