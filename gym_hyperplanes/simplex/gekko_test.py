from gekko import GEKKO


def test1():
    m = GEKKO()  # Initialize gekko
    # Initialize variables
    x1 = m.Var(value=0, lb=0, ub=100)
    x2 = m.Var(value=0, lb=0, ub=100)

    m.Equation(1 * x1 + 1 * x2 >= 1.5)
    m.Equation(1 * x1 + 1 * x2 <= 4.5)
    m.Equation(-1 * x1 + 1 * x2 >= -1)
    m.Equation(-1 * x1 + 1 * x2 <= 3)
    m.Obj(x1 * x1 + x2 * x2)  # Objective
    m.solve(disp=False)  # Solve
    print("x1: " + str(x1.value))
    print("x2: " + str(x2.value))
    print("Objective: " + str(m.options.objfcnval))


def test2():
    m = GEKKO()  # Initialize gekko
    # Initialize variables
    x1 = m.Var(value=0, lb=0, ub=100)
    x2 = m.Var(value=0, lb=0, ub=100)

    m.Equation(0.25881904510252124 * x1 + -0.9659258262890682 * x2 >= 0)
    m.Equation(0.9659258262890682 * x1 + 0.25881904510252096 * x2 >= 25)
    m.Equation(0.5000000000000001 * x1 + 0.8660254037844386 * x2 >= 40)
    m.Equation(0.8660254037844386 * x1 + 0.5000000000000001 * x2 >= 60)
    m.Obj(x1 * x1 + x2 * x2)  # Objective
    m.solve(disp=False)  # Solve
    print("x1: " + str(x1.value))
    print("x2: " + str(x2.value))
    print("Objective: " + str(m.options.objfcnval))


def main():
    test2()


if __name__ == "__main__":
    main()
