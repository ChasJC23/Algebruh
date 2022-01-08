from parser import *
from tokeniser import Tokeniser

def main():
    tree = Parser.parse("3 ** x / 2 ** x")
    print(tree)
    simplified = tree.simplify()
    while tree != simplified:
        print(simplified)
        tree = simplified
        simplified = tree.simplify()

if __name__ == "__main__":
    main()