from abc import ABC, abstractmethod
import cmath
from typing import Callable, Iterable
from .nodes import Node, FunctionCallNode

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
    def differentiateFtn(self, function: str, *args: Node) -> Iterable[Node]:
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

    def differentiateFtn(self, function: str, *args: Node) -> Iterable[Node]:
        match function:
            case "sqrt": return (1 / (2 * FunctionCallNode("sqrt", args[0])),)
            case "rect": return (FunctionCallNode("exp", args[1]), FunctionCallNode("rect", args[0], args[1]))
            case "exp": return (FunctionCallNode("exp", args[0]),)
            case "ln" | "log": return (1 / args[0],)
            case "log10": return (1 / (args[0] * FunctionCallNode("log", 10)),)
            case "sin": return (FunctionCallNode("cos", args[0]),)
            case "cos": return (-FunctionCallNode("sin", args[0]),)
            case "tan": return (FunctionCallNode("sec", args[0]) ** 2,)
            case "sec": return (FunctionCallNode("tan", args[0]) * FunctionCallNode("sec", args[0]),)
            case "csc": return (-(FunctionCallNode("cot", args[0]) * FunctionCallNode("csc", args[0])),)
            case "cot": return (-FunctionCallNode("csc", args[0]) ** 2,)
            case "asin": return (1 / FunctionCallNode("sqrt", 1 - args[0] ** 2),)
            case "acos": return (-1 / FunctionCallNode("sqrt", 1 - args[0] ** 2),)
            case "atan": return (1 / (1 + args[0] ** 2),)
            case "sinh": return (FunctionCallNode("cosh", args[0]))
            case "cosh": return (FunctionCallNode("sinh", args[0]))
            case "tanh": return (FunctionCallNode("sech", args[0]) ** 2,)
            case "sech": return (-(FunctionCallNode("tanh", args[0]) * FunctionCallNode("sech", args[0])),)
            case "csch": return (-(FunctionCallNode("coth", args[0]) * FunctionCallNode("csch", args[0])),)
            case "coth": return (-FunctionCallNode("csch", args[0]) ** 2,)
            case "asinh": return (1 / FunctionCallNode("sqrt", args[0] ** 2 + 1),)
            case "acosh": return (1 / FunctionCallNode("sqrt", (args[0] ** 2 - 1) * (args[0] ** 2 + 1)),)
            case "atanh": return (1 / (1 - args[0] ** 2),)