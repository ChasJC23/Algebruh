# Mathematical Expression System

System for parsing, simplifying, differentiating, and evaluating mathematical expressions.

```py
from algebruh import expression, StdContext

expr = expression("3 * x ** 2 - 7 * x + 2")
assert(expr.evaluate(StdContext(x=2)) == 0)

derivative = expr.differentiate(StdContext(), "x")

# taking the derivative of an expression uses standard laws of differentiation,
# so it can produce an expression of an unusual form.
print(derivative)
# all we need to do to fix that is simplify the expression!
derivative = derivative.simplify()
print(derivative)

assert(derivative.evaluate(StdContext(x=1)) == -1)

# let's try and do some Newton-Raphson to try and find the other solution.
x = 1
# we could simplify the ratio, however the algorithm isn't yet smart enough to know how to manipulate fractions.
ratio = expr / derivative
for i in range(5):
  x = x - ratio.evaluate(StdContext(x=x))
  print(x)

# but what if our variable x respresented something a bit more complicated?
# well we can substitute it with another expression
expr_for_x = expression("sin(t) + t")
expr = expr.substitute("x", expr_for_x)
print(expr)
print(expr.evaluate(StdContext(t=3)))
```

## Improvements to-do:

- [ ] Design algorithm for product simplification
- [ ] Expression factorisation for polynomials
- [ ] Use of functional identities for expression simplification
- [x] Implement operators to allow expressions to be extended through simple arithmetic.
- [x] Fix up inconsistencies within `__str__` and `__repr__`
- [x] Implement derivatives for the remaining functions
- [x] Upgrade variable substitution to expression substitution
- [ ] Allow non-parsed expressions in expression substitution
- [ ] Replace partial expression with substitution
- [ ] Supply multiple different numerical types
- [ ] Hide node type from user, use `Expression` alias instead(?)