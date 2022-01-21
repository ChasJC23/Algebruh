from enum import IntEnum
from io import StringIO
from string import whitespace as WHITESPACE, digits as DIGITS, ascii_letters as LETTERS

class Token(IntEnum):
    EOF = 0
    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    LPARENS = 5
    RPARENS = 6
    POW = 7
    LITERAL = 8
    IDENTIFIER = 9
    SEPARATOR = 10

class Tokeniser:
    '''
    class responsible for tokenising a given mathematical expression ready to be parsed.
    '''
    def __init__(self, stream: StringIO):
        self.currentChar: str = None
        self.currentToken: Token = None
        self.litValue: complex = None
        self.identifier: str = None
        self.stream = stream
        self.nextChar()
        self.nextToken()
    
    def nextChar(self):
        self.currentChar = self.stream.read(1)
    
    def nextToken(self):
        while self.currentChar in WHITESPACE and self.currentChar != '':
            self.nextChar()
        match self.currentChar:
            case '+':
                self.currentToken = Token.ADD
                self.nextChar()
                return
            case '-':
                self.currentToken = Token.SUB
                self.nextChar()
                return
            case '*':
                self.nextChar()
                if self.currentChar == '*':
                    self.currentToken = Token.POW
                    self.nextChar()
                else: self.currentToken = Token.MUL
                return
            case '/':
                self.currentToken = Token.DIV
                self.nextChar()
                return
            case '(':
                self.currentToken = Token.LPARENS
                self.nextChar()
                return
            case ')':
                self.currentToken = Token.RPARENS
                self.nextChar()
                return
            case ',':
                self.currentToken = Token.SEPARATOR
                self.nextChar()
                return
            case '':
                self.currentToken = Token.EOF
                return
        if self.currentChar in DIGITS + '.' and self.currentChar != '':
            litDigits = ""
            seenRadix = False
            while (self.currentChar in DIGITS or self.currentChar == '.' and not seenRadix) and self.currentChar != '':
                seenRadix |= self.currentChar == '.'
                litDigits += self.currentChar
                self.nextChar()
            self.litValue = float(litDigits) if seenRadix else int(litDigits)
            if self.currentChar == 'j':
                self.nextChar()
                self.litValue *= 1j
            self.currentToken = Token.LITERAL
            return
        if self.currentChar in LETTERS and self.currentChar != '':
            self.identifier = ""
            while self.currentChar in LETTERS + DIGITS and self.currentChar != '':
                self.identifier += self.currentChar
                self.nextChar()
            self.currentToken = Token.IDENTIFIER
            return