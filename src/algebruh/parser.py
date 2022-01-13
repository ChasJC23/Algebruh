from tokeniser import Token, Tokeniser
from io import StringIO
from nodes import *

class Parser:
    '''
    class responsible for parsing a given mathematical expression within a tokeniser.
    '''
    def __init__(self, tokeniser: Tokeniser):
        self.tokeniser = tokeniser
    
    def parseExpression(self) -> Node:
        tree = self.parseAddSub()
        if self.tokeniser.currentToken != Token.EOF:
            raise SyntaxError()
        return tree

    def parseAddSub(self) -> Node:
        lhs = self.parseMulDiv()
        while True:
            token = self.tokeniser.currentToken
            if token not in (Token.ADD, Token.SUB):
                return lhs
            self.tokeniser.nextToken()
            rhs = self.parseMulDiv()
            lhs = AddNode(lhs, rhs) if token == Token.ADD else SubNode(lhs, rhs)
    
    def parseMulDiv(self) -> Node:
        lhs = self.parsePow()
        while True:
            token = self.tokeniser.currentToken
            if token not in (Token.MUL, Token.DIV):
                return lhs
            self.tokeniser.nextToken()
            rhs = self.parsePow()
            lhs = MulNode(lhs, rhs) if token == Token.MUL else DivNode(lhs, rhs)

    def parsePow(self) -> Node:
        lhs = self.parsePosNeg()
        if self.tokeniser.currentToken == Token.POW:
            self.tokeniser.nextToken()
            rhs = self.parsePow()
            return PowNode(lhs, rhs)
        return lhs

    def parsePosNeg(self) -> Node:
        match self.tokeniser.currentToken:
            case Token.ADD:
                self.tokeniser.nextToken()
                return PosNode(self.parsePosNeg())
            case Token.SUB:
                self.tokeniser.nextToken()
                return NegNode(self.parsePosNeg())
            case _:
                return self.parseLeaf()

    def parseLeaf(self) -> Node:
        expr: Node
        if self.tokeniser.currentToken == Token.LPARENS:
            self.tokeniser.nextToken()
            expr = self.parseAddSub()
            if self.tokeniser.currentToken != Token.RPARENS: raise SyntaxError()
            self.tokeniser.nextToken()
        elif self.tokeniser.currentToken == Token.LITERAL:
            expr = LiteralNode(self.tokeniser.litValue)
            self.tokeniser.nextToken()
        elif self.tokeniser.currentToken == Token.IDENTIFIER:
            identifier = self.tokeniser.identifier
            self.tokeniser.nextToken()
            if self.tokeniser.currentToken == Token.LPARENS:
                arguments: list[Node] = []
                while True:
                    self.tokeniser.nextToken()
                    arguments.append(self.parseAddSub())
                    if self.tokeniser.currentToken == Token.SEPARATOR:
                        self.tokeniser.nextToken()
                    elif self.tokeniser.currentToken == Token.RPARENS:
                        self.tokeniser.nextToken()
                        return FunctionCallNode(identifier, *arguments)
                    else:
                        raise SyntaxError()
            else:
                expr = VariableNode(identifier)
        else:
            raise SyntaxError()
        return expr
    @staticmethod
    def parse(expression: StringIO | str):
        '''
        directly parses the given mathematical expression.
        '''
        if isinstance(expression, str):
            expression = StringIO(expression)
        return Parser(Tokeniser(expression)).parseExpression()