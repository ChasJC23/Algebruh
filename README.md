# Mathematical Expression System

System for parsing, simplifying, differentiating, and evaluating mathematical expressions.

```py
from algebruh import expression

expr = expression("3 * x ** 2 - 7 * x + 2")
assert expr.evaluate(x=2) == 0

derivative = expr.differentiate("x")

# taking the derivative of an expression uses standard laws of differentiation,
# so it can produce an expression of an unusual form.
print(derivative)
# all we need to do to fix that is simplify the expression!
derivative = derivative.simplify()
print(derivative)

assert derivative.evaluate(x=1) == -1

# let's try and do some Newton-Raphson to try and find the other solution.
x = 1
# we could simplify the ratio, however the algorithm isn't yet smart enough to know how to manipulate fractions.
ratio = expr / derivative
for i in range(5):
    x = x - ratio.evaluate(x=x)
    print(x)

# but what if our variable x respresented something a bit more complicated?
# well we can substitute it with another expression
expr = expr.substitute("x", "sin(t) + t")
print(expr)
print(expr.evaluate(t=3))
```