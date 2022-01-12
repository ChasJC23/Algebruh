# Mathematical Expression System

System for parsing, simplifying, differentiating, and evaluating mathematical expressions.

```py
expression = Parser.parse("3 * x ** 2 - 7 * x + 2")
assert(expression.evaluate(StdContext(x=2)) == 0)

derivative = expression.differentiate()

# taking the derivative of an expression uses standard laws of differentiation,
# so it can produce an expression of an unusual form.
print(derivative)
# all we need to do to fix that is simplify the expression!
print(derivative.simplify())

assert(derivative.evaluate(StdContext(x=1)) == -1)
```

## Improvements to-do:

- [ ] Design algorithm for product simplification
- [ ] Expression factorisation for polynomials
- [ ] Use of functional identities for expression simplification
- [ ] Implement operators to allow expressions to be extended through simple arithmetic.
- [ ] Fix up inconsistencies within `__str__` and `__repr__`
- [ ] Implement derivatives for the remaining functions