from nodes import StdContext
from parser import Parser


expression = Parser.parse("3 ** x / 2 ** x")

derivative = expression.differentiate(StdContext(), "x")

print(derivative)
print(derivative.simplify())