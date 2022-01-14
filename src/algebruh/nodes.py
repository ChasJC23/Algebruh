from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generator, Callable, TypeVar, Generic
from copy import copy
from . import *

T = TypeVar('T')

class PartialExpression(Callable):
    '''
    callable class holding an incomplete expression which can be completed when called.
    '''
    def __init__(self, tree: Node):
        self.tree = tree

    def __call__(self, *args: Node) -> Node:
        for node in walk(self.tree):
            if isinstance(node, BinaryNode):
                if isinstance(node.left, PlaceholderNode):
                    node.left = args[node.left.id]
                if isinstance(node.right, PlaceholderNode):
                    node.right = args[node.right.id]
            if isinstance(node, UnaryNode):
                if isinstance(node.arg, PlaceholderNode):
                    node.arg = args[node.arg.id]
            if isinstance(node, FunctionCallNode):
                for i, arg in enumerate(node.arguments):
                    if isinstance(arg, PlaceholderNode):
                        node.arguments[i] = args[arg.id]
        return self.tree

def walk(tree: Node) -> Generator[Node, None, None]:
    '''
    depth-first walk through an expression
    '''
    match tree:
        case BinaryNode():
            for i in walk(tree.left):
                yield i
            for i in walk(tree.right):
                yield i
        case UnaryNode():
            for i in walk(tree.arg):
                yield i
        case FunctionCallNode():
            for arg in tree.arguments:
                for i in walk(arg):
                    yield i
    yield tree

def sumprod_terms(tree: AddSubNode | MulDivNode) -> Generator[tuple[Node, bool], None, None]:
    '''
    yields the terms within the sum or product as determined by the given tree,
    alongside a boolean value indicating if each are negated / inversed.
    '''
    root_type = AddSubNode if isinstance(tree, AddSubNode) else MulDivNode if isinstance(tree, MulDivNode) else None
    if root_type == None: raise TypeError()
    if isinstance(tree.left, root_type):
        for term, neg in sumprod_terms(tree.left):
            yield term, neg ^ tree.left_negated
    else:
        yield tree.left, tree.left_negated
    if isinstance(tree.right, root_type):
        for term, neg in sumprod_terms(tree.right):
            yield term, neg ^ tree.right_negated
    else:
        yield tree.right, tree.right_negated

class Node(ABC):
    '''
    base node class all expression elements are based on.
    '''
    @abstractmethod
    def evaluate(self, context: Context) -> complex:
        '''
        returns the value of this node as determined by a specific context.
        '''
    @abstractmethod
    def differentiate(self, context: Context, var: str) -> Node:
        '''
        returns a new node representing the node's derivative with respect to a given variable.
        '''

    def simplify(self: T, callerType: type = object) -> T:
        '''
        returns a new node which includes less redundant operations.
        '''
        return copy(self)
        
    def substitute(self: T, var: str | Node, expr: Node) -> T:
        '''
        substitutes all instances of a variable or expression with a new expression.
        '''
        return copy(self)
    
    def __hash__(self) -> int:
        return hash(self.__repr__())
    
    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__)
    
    def __pos__(self) -> PosNode:
        return PosNode(self)
    def __neg__(self) -> NegNode:
        return NegNode(self)
    def __add__(self, other: Node | complex) -> AddNode:
        if isinstance(other, Node): return AddNode(self, other)
        elif isinstance(other, str): return AddNode(self, VariableNode(other))
        else: return AddNode(self, LiteralNode(other))
    def __radd__(self, other: str | complex) -> AddNode:
        if isinstance(other, str): return AddNode(VariableNode(other), self)
        else: return AddNode(LiteralNode(other), self)
    def __sub__(self, other: Node | complex) -> SubNode:
        if isinstance(other, Node): return SubNode(self, other)
        elif isinstance(other, str): return SubNode(self, VariableNode(other))
        else: return SubNode(self, LiteralNode(other))
    def __rsub__(self, other: str | complex) -> SubNode:
        if isinstance(other, str): return SubNode(VariableNode(other), self)
        else: return SubNode(LiteralNode(other), self)
    def __mul__(self, other: Node | complex) -> MulNode:
        if isinstance(other, Node): return MulNode(self, other)
        elif isinstance(other, str): return MulNode(self, VariableNode(other))
        else: return MulNode(self, LiteralNode(other))
    def __rmul__(self, other: str | complex) -> MulNode:
        if isinstance(other, str): return MulNode(VariableNode(other), self)
        else: return MulNode(LiteralNode(other), self)
    def __truediv__(self, other: Node | complex) -> DivNode:
        if isinstance(other, Node): return DivNode(self, other)
        elif isinstance(other, str): return DivNode(self, VariableNode(other))
        else: return DivNode(self, LiteralNode(other))
    def __rtruediv__(self, other: str | complex) -> DivNode:
        if isinstance(other, str): return DivNode(VariableNode(other), self)
        else: return DivNode(LiteralNode(other), self)
    def __pow__(self, other: Node | complex) -> PowNode:
        if isinstance(other, Node): return PowNode(self, other)
        elif isinstance(other, str): return PowNode(self, VariableNode(other))
        else: return PowNode(self, LiteralNode(other))
    def __rpow__(self, other: str | complex) -> PowNode:
        if isinstance(other, str): return PowNode(VariableNode(other), self)
        else: return PowNode(LiteralNode(other), self)

class PlaceholderNode(Node, Generic[T]):
    '''
    element of an expression with an unknown value to be replaced when put into a partial expression.
    '''
    def __init__(self, _id: T) -> None:
        self.id = _id

    def __eq__(self, __o: PlaceholderNode) -> bool:
        return super().__eq__(__o) and self.id == __o.id
    
    def __hash__(self) -> int:
        return super().__hash__()

class LiteralNode(Node):
    '''
    expression element representing a literal value.
    '''
    def __init__(self, value: complex):
        self.value = value

    def evaluate(self, context: Context) -> complex:
        return self.value

    def differentiate(self, context: Context, var: str) -> Node:
        return LiteralNode(0)

    def __repr__(self) -> str:
        return f"Lit{self.value}"
    
    def __str__(self) -> str:
        return str(self.value)
    
    def __hash__(self) -> int:
        return super().__hash__()
    
    def __eq__(self, __o: LiteralNode) -> bool:
        return super().__eq__(__o) and self.value == __o.value

class VariableNode(Node):
    '''
    expression element representing an unknown variable in an expression.
    '''
    def __init__(self, identifier: str):
        self.identifier = identifier

    def evaluate(self, context: Context) -> complex:
        return context.resolveVariable(self.identifier)

    def differentiate(self, context: Context, var: str) -> Node:
        if var == self.identifier: return LiteralNode(1)
        else: return LiteralNode(0)
    
    def substitute(self, var: str | Node, expr: Node) -> Node:
        return expr if isinstance(var, str) and self.identifier == var or self == var else super().substitute(var, expr)

    def __repr__(self) -> str:
        return f"Var({self.identifier})"
    
    def __str__(self) -> str:
        # redundant operation considering the type of the identifier should be string,
        # but it resolves to another __str__ call so ¯\_(ツ)_/¯
        return str(self.identifier)
    
    def __hash__(self) -> int:
        return super().__hash__()
    
    def __eq__(self, __o: VariableNode) -> bool:
        return super().__eq__(__o) and self.identifier == __o.identifier

class FunctionCallNode(Node):
    '''
    expression element representing a call to a specific function.
    The function may be multi-valued.
    '''
    def __init__(self, identifier: str, *arguments: Node):
        self.identifier = identifier
        self.arguments = list(arguments)

    def evaluate(self, context: Context) -> complex:
        return context.resolveFunction(self.identifier)(*(arg.evaluate(context) for arg in self.arguments))

    def differentiate(self, context: Context, var: str) -> Node:
        fprimeconstructors = context.differentiateFtn(self.identifier)
        # fprimenode = FunctionCallNode(fprimeftn, *self.arguments)
        result: Node = None
        for i in range(len(fprimeconstructors)):
            argDeriv = self.arguments[i].differentiate(context, var)
            constructor: PartialExpression = fprimeconstructors[i]
            fprime = constructor(*self.arguments)
            term = MulNode(fprime, argDeriv)
            if result is None: result = term
            else: result = AddNode(result, term)
        return result or LiteralNode(0)

    def simplify(self: T, callerType: type = object) -> T:
        nc = super().simplify(callerType)
        for i, arg in enumerate(nc.arguments):
            nc.arguments[i] = arg.simplify(self.__class__)
        return nc
    
    def substitute(self, var: str | Node, expr: Node) -> Node:
        if isinstance(var, Node) and self == var: return expr
        nc = super().substitute(var, expr)
        for i, arg in enumerate(nc.arguments):
            nc.arguments[i] = arg.substitute(self.__class__, var, expr)
        return nc

    def __repr__(self) -> str:
        return f"Ftn[{self.identifier}]{self.arguments.__repr__()}"
    
    def __str__(self) -> str:
        return self.identifier + (f'({self.arguments[0]})' if len(self.arguments) == 1 else str(self.arguments))
    
    def __hash__(self) -> int:
        return super().__hash__()
    
    def __eq__(self, __o: FunctionCallNode) -> bool:
        return super().__eq__(__o) and self.identifier == __o.identifier and all(self.arguments[i] == __o.arguments[i] for i in range(len(self.arguments)))

class UnaryNode(Node, ABC):
    '''
    expression element corresponding to any unary operator.
    '''
    def __init__(self, op: Callable[[complex], complex], arg: Node):
        self.op = op
        self.arg = arg

    def evaluate(self, context: Context) -> complex:
        return self.op(self.arg.evaluate(context))

    def simplify(self, callerType: type = object) -> UnaryNode:
        nc = super().simplify(callerType)
        nc.arg = self.arg.simplify(self.__class__)
        return nc
    
    def substitute(self, var: str | Node, expr: Node) -> UnaryNode:
        if isinstance(var, Node) and self == var: return expr
        nc = super().substitute(var, expr)
        nc.arg = self.arg.substitute(var, expr)
        return nc
    
    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, __o: UnaryNode) -> bool:
        return super().__eq__(__o) and self.arg == __o.arg

class PosNode(UnaryNode):
    '''
    expression element representing the `+a` prefix unary operator.
    '''
    def __init__(self, arg: Node):
        op = lambda x: +x
        super().__init__(op, arg)

    def differentiate(self, context: Context, var: str) -> Node:
        return self.arg.differentiate(context, var)

    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        return nc.arg

    def __repr__(self) -> str:
        return f"(+{self.arg.__repr__()})"

    def __str__(self) -> str:
        return f"(+{self.arg})"

    def __hash__(self) -> int:
        return super().__hash__()

class NegNode(UnaryNode):
    '''
    expression element representing the `-a` prefix unary operator.
    '''
    def __init__(self, arg: Node):
        op = lambda x: -x
        super().__init__(op, arg)

    def differentiate(self, context: Context, var: str) -> Node:
        return NegNode(self.arg.differentiate(context, var))

    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        if isinstance(nc.arg, NegNode):
            return nc.arg.arg
        elif isinstance(nc.arg, LiteralNode):
            return LiteralNode(-nc.arg.value)
        else:
            return nc

    def __repr__(self) -> str:
        return f"(-{self.arg.__repr__()})"

    def __str__(self) -> str:
        return f"(-{self.arg})"

    def __hash__(self) -> int:
        return super().__hash__()

class BinaryNode(Node, ABC):
    '''
    expression element corresponding to any binary operator.
    '''
    def __init__(self, op: Callable[[complex, complex], complex], left: Node, right: Node, negations: int):
        self.op = op
        self.left = left
        self.right = right
        self.negations = negations
    @property
    def left_negated(self) -> bool:
        return bool(self.negations & 0b10)
    @property
    def right_negated(self) -> bool:
        return bool(self.negations & 0b01)

    def evaluate(self, context: Context) -> complex:
        return self.op(self.left.evaluate(context), self.right.evaluate(context))

    def simplify(self: T, callerType: type = object) -> T:
        nc = super().simplify(callerType)
        nc.left = nc.left.simplify(self.__class__)
        nc.right = nc.right.simplify(self.__class__)
        return nc
    
    def substitute(self, var: str | Node, expr: Node) -> BinaryNode:
        if isinstance(var, Node) and self == var: return expr
        nc = super().substitute(var, expr)
        nc.left = self.left.substitute(var, expr)
        nc.right = self.right.substitute(var, expr)
        return nc
    
    def __hash__(self) -> int:
        return super().__hash__()

    def __eq__(self, __o: BinaryNode) -> bool:
        return super().__eq__(__o) and (self.left == __o.left and self.right == __o.right or self.left == __o.right and self.right == __o.left)

class AddSubNode(BinaryNode, ABC):
    '''
    expression element representing either binary addition or subtraction. 
    '''
    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        if issubclass(callerType, AddSubNode) and isinstance(nc, AddSubNode): return nc
        literalAmount = 0
        variableAmount: dict[Node, complex] = {}
        for term, negated in sumprod_terms(nc):
            if isinstance(term, LiteralNode):
                literalAmount += -term.value if negated else term.value
                continue
            quantity = 1
            match term:
                case MulNode():
                    if isinstance(term.left, LiteralNode):
                        quantity = term.left.value; term = term.right
                    elif isinstance(term.right, LiteralNode):
                        quantity = term.right.value; term = term.left
                case DivNode():
                    if isinstance(term.left, LiteralNode):
                        quantity = term.left.value; term = term.right
                    elif isinstance(term.right, LiteralNode):
                        quantity = 1/term.right.value; term = term.left
            if term in variableAmount.keys():
                variableAmount[term] += -quantity if negated else quantity
            else:
                variableAmount[term] = -quantity if negated else quantity
        if variableAmount == {}:
            return LiteralNode(literalAmount)
        else:
            return sumFromDict(variableAmount, literalAmount)

    def differentiate(self, context: Context, var: str):
        return self.__class__(self.left.differentiate(context, var), self.right.differentiate(context, var))

def sumFromDict(d: dict[Node, complex], literal: complex = 0):
    '''
    returns a sum of all keys in the dictionary, where their coefficient is their linked value in the dictionary.
    '''
    dc = copy(d)
    expr = LiteralNode(0)
    while expr == LiteralNode(0) and dc != {}:
        key, value = dc.popitem()
        expr = MulNode(LiteralNode(value), key).simplify()
    if dc != {}:
        for key, value in dc.items():
            term = MulNode(LiteralNode(value), key).simplify()
            if term != LiteralNode(0):
                expr = AddNode(expr, term)
    if literal != 0:
        if isinstance(expr, LiteralNode):
            expr.value += literal
        else:
            expr = AddNode(expr, LiteralNode(literal))
    return expr

def productFromDict(d: dict[Node, Node], literal: complex = 1):
    '''
    returns a product of all keys in the dictionary, where their exponent is their linked value in the dictionary.
    '''
    if literal == 0: return LiteralNode(0)
    dc = copy(d)
    expr = LiteralNode(1)
    while expr == LiteralNode(1) and dc != {}:
        key, value = dc.popitem()
        expr = PowNode(key, value).simplify()
    if dc != {}:
        for key, value in dc.items():
            term = PowNode(key, value).simplify()
            if isinstance(term, LiteralNode):
                literal *= term.value
            else:
                expr = MulNode(expr, term)
    if literal != 1:
        if isinstance(expr, LiteralNode):
            expr.value *= literal
        else:
            expr = MulNode(LiteralNode(literal), expr)
    return expr

class AddNode(AddSubNode):
    '''
    expression element representing the binary addition operator. 
    '''
    def __init__(self, left: Node, right: Node):
        op = lambda x, y: x + y
        super().__init__(op, left, right, 0b00)

    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        if not isinstance(nc, self.__class__): return nc
        # 0 + x = x
        elif isinstance(nc.left, LiteralNode) and nc.left.value == 0:
            return nc.right
        # x + 0 = x
        elif isinstance(nc.right, LiteralNode) and nc.right.value == 0:
            return nc.left
        # x + -y = x - y
        elif isinstance(nc.right, NegNode):
            return SubNode(nc.left, nc.right.arg)
        else:
            return nc

    def __repr__(self) -> str:
        return f"({self.left.__repr__()} + {self.right.__repr__()})"
    
    def __str__(self) -> str:
        return f"({self.left} + {self.right})"

class SubNode(AddSubNode):
    '''
    expression element representing the binary subtraction operator. 
    '''
    def __init__(self, left: Node, right: Node):
        op = lambda x, y: x - y
        super().__init__(op, left, right, 0b01)

    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        if not isinstance(nc, self.__class__): return nc
        elif isinstance(nc.left, LiteralNode) and nc.left.value == 0:
            return NegNode(nc.right)
        elif isinstance(nc.right, LiteralNode) and nc.right.value == 0:
            return nc.left
        elif isinstance(nc.right, NegNode):
            return SubNode(nc.left, nc.right.arg)
        else:
            return nc

    def __repr__(self) -> str:
        return f"({self.left.__repr__()} - {self.right.__repr__()})"
    
    def __str__(self) -> str:
        return f"({self.left} - {self.right})"

class MulDivNode(BinaryNode, ABC):
    '''
    expression element representing either binary multiplication or division. 
    '''
    def simplify(self, callerType: type = object) -> BinaryNode:
        nc = super().simplify(callerType)
        return nc

class MulNode(MulDivNode):
    '''
    expression element representing the binary multiplication operator. 
    '''
    def __init__(self, left: Node, right: Node):
        op = lambda x, y: x * y
        super().__init__(op, left, right, 0b00)

    def differentiate(self, context: Context, var: str) -> Node:
        # product rule
        return AddNode(
            MulNode(self.left, self.right.differentiate(context, var)),
            MulNode(self.left.differentiate(context, var), self.right)
        )

    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        if not isinstance(nc, self.__class__): return nc
        # 1 * x = x, x * 0 = 0
        elif isinstance(nc.left, LiteralNode) and nc.left.value == 1 or isinstance(nc.right, LiteralNode) and nc.right.value == 0:
            return nc.right
        # x * 1 = x, 0 * x = 0
        elif isinstance(nc.right, LiteralNode) and nc.right.value == 1 or isinstance(nc.left, LiteralNode) and nc.left.value == 0:
            return nc.left
        # x * (1 / y) = x / y
        elif isinstance(nc.right, DivNode) and isinstance(nc.right.left, LiteralNode) and nc.right.left.value == 1:
            return DivNode(nc.left, nc.right.right)
        elif isinstance(nc.right, DivNode) and isinstance(nc.right.left, LiteralNode) and nc.right.left.value == -1:
            return NegNode(DivNode(nc.left, nc.right.right))
        # (1 / y) * x = x / y
        elif isinstance(nc.left, DivNode) and isinstance(nc.left.left, LiteralNode) and nc.left.left.value == 1:
            return DivNode(nc.left.right, nc.right)
        elif isinstance(nc.left, DivNode) and isinstance(nc.left.left, LiteralNode) and nc.left.left.value == -1:
            return NegNode(DivNode(nc.left.right, nc.right))
        else:
            return nc

    def __repr__(self) -> str:
        return f"({self.left.__repr__()} * {self.right.__repr__()})"
    
    def __str__(self) -> str:
        return f"({self.left} * {self.right})"

class DivNode(MulDivNode):
    '''
    expression element representing the binary division operator. 
    '''
    def __init__(self, left: Node, right: Node):
        op = lambda x, y: x / y
        super().__init__(op, left, right, 0b01)

    def differentiate(self, context: Context, var: str) -> Node:
        # quotient rule
        return DivNode(
            SubNode(
                MulNode(self.left.differentiate(context, var), self.right),
                MulNode(self.left, self.right.differentiate(context, var))
            ),
            MulNode(self.right, self.right)
        )

    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        if not isinstance(nc, self.__class__): return nc
        # x / 1 = x, 0 / x = 0
        elif isinstance(nc.right, LiteralNode) and nc.right == 1 or isinstance(nc.left, LiteralNode) and nc.left == 0:
            return nc.left
        else:
            return nc

    def __repr__(self) -> str:
        return f"({self.left.__repr__()} / {self.right.__repr__()})"
    
    def __str__(self) -> str:
        return f"({self.left} / {self.right})"

class PowNode(BinaryNode):
    '''
    expression element representing the binary exponent operator. 
    '''
    def __init__(self, left: Node, right: Node):
        op = lambda x, y: x ** y
        super().__init__(op, left, right, 0b00)

    def differentiate(self, context: Context, var: str) -> Node:
        # oh god, exponents are so much worse than you think
        return MulNode(
            PowNode(self.left, SubNode(self.right, LiteralNode(1))),
            AddNode(
                # NOTE: I have no idea how commutativity plays into this, I'm just trusting Wolfram Alpha :)
                MulNode(self.right, self.left.differentiate(context, var)),
                MulNode(
                    MulNode(
                        self.left,
                        FunctionCallNode("ln", self.left)
                    ),
                    self.right.differentiate(context, var)
                )
            )
        )

    def simplify(self, callerType: type = object) -> Node:
        nc = super().simplify(callerType)
        if callerType != self.__class__:
            while isinstance(nc.left, PowNode):
                nc.right = MulNode(nc.left.right, nc.right)
                nc.left = nc.left.left
        if isinstance(nc.left, LiteralNode) and isinstance(nc.right, LiteralNode):
            return LiteralNode(nc.left.value ** nc.right.value)
        # 1 ** x = 1, x ** 1 = x
        elif isinstance(nc.left, LiteralNode) and nc.left.value == 1 or isinstance(nc.right, LiteralNode) and nc.right.value == 1:
            return nc.left
        # (x ** y) ** z = x ** (y * z)
        elif isinstance(nc.left, PowNode):
            return PowNode(nc.left.left, MulNode(nc.left.right, nc.right))
        else:
            return nc

    def __repr__(self) -> str:
        return f"({self.left.__repr__()} ** {self.right.__repr__()})"
    
    def __str__(self) -> str:
        return f"({self.left} ** {self.right})"
