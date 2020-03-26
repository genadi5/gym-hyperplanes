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


def test3():  # 11101
    m = GEKKO(remote=False)  # Initialize gekko
    # Initialize variables
    x1 = m.Var(value=0, lb=0, ub=100)
    x2 = m.Var(value=0, lb=0, ub=100)

    m.Equation(0 * x1 + 1 * x2 >= 0)
    m.Equation(-0.25881904510252124 * x1 + 0.9659258262890682 * x2 <= 0)
    m.Equation(0.9659258262890682 * x1 + 0.25881904510252096 * x2 >= 25)
    m.Equation(0.5000000000000001 * x1 + 0.8660254037844386 * x2 >= 40)
    m.Equation(0.8660254037844386 * x1 + 0.5000000000000001 * x2 >= 60)
    m.Obj(x1 * x1 + x2 * x2)  # Objective
    m.solve(disp=False)  # Solve
    print("x1: " + str(x1.value))
    print("x2: " + str(x2.value))
    print("Objective: " + str(m.options.objfcnval))


def test4():  # 01111
    m = GEKKO(remote=False)  # Initialize gekko
    # Initialize variables
    x1 = m.Var(value=0, lb=0, ub=100)
    x2 = m.Var(value=0, lb=0, ub=100)

    m.Equation(0 * x1 + 1 * x2 >= 0)
    m.Equation(-0.25881904510252124 * x1 + 0.9659258262890682 * x2 >= 0)
    m.Equation(0.9659258262890682 * x1 + 0.25881904510252096 * x2 >= 25)
    m.Equation(0.5000000000000001 * x1 + 0.8660254037844386 * x2 >= 40)
    m.Equation(0.8660254037844386 * x1 + 0.5000000000000001 * x2 <= 60)
    m.Obj(x1 * x1 + x2 * x2)  # Objective
    m.solve(disp=False)  # Solve
    print("x1: " + str(x1.value))
    print("x2: " + str(x2.value))
    print("Objective: " + str(m.options.objfcnval))


def test5():
    m = GEKKO(remote=False)  # Initialize gekko
    # Initialize variables
    x1 = m.Var(value=0, lb=0, ub=10)
    x2 = m.Var(value=0, lb=0, ub=10)
    x3 = m.Var(value=0, lb=0, ub=10)
    x4 = m.Var(value=0, lb=0, ub=10)

    m.Equation(-0.77 * x1 - 0.64 * x2 - 0 * x3 + 0 * x4 < 6.3)
    m.Equation(0 * x1 - 0.11 * x2 - 0 * x3 + 0.99 * x4 >= -0.315)
    m.Equation(-0.43 * x1 - 0.3132 * x2 - 0.431 * x3 - 0.73 * x4 < 0)
    m.Equation(0.15 * x1 + 0.15 * x2 + 0.15 * x3 + 0.966 * x4 < 3.465)
    m.Equation(0.593 * x1 + 0.44776 * x2 + 0.31 * x3 + 0.59 * x4 >= 5.985)

    m.Equation(x1 >= 0)
    m.Equation(x2 >= 0)
    m.Equation(x3 >= 0)
    m.Equation(x4 > 0)
    m.Obj((x1 - 4.4) * (x1 - 4.4) + (x2 - 2.9) * (x2 - 2.9) + (x3 - 1.4) * (x3 - 1.4) + (x4 - 0.2) * (x4 - 0.2))
    m.solve(disp=False)  # Solve
    print("x1: " + str(x1.value))
    print("x2: " + str(x2.value))
    print("Objective: " + str(m.options.objfcnval))

    # Eq: [(((((-0.769517879287109) * (v1)) + ((-0.6386252684144829) * (v2))) + ((0.0) * (v3))) + (
    #             (0.0) * (v4))) >= 6.3000000000000025]
    # Eq: [(((((-2.1841450407278862e-09) * (v1)) + ((-0.11073801741243763) * (v2))) + (
    #             (-2.1841450407278862e-09) * (v3))) + ((0.9938496322379773) * (v4))) < -0.31499999999999967]
    # Eq: [(((((-0.43089933533778474) * (v1)) + ((-0.3132017009541168) * (v2))) + ((-0.43089933533778474) * (v3))) + (
    #             (-0.7283929023064123) * (v4))) >= 3.3306690738754696e-16]
    # Eq: [(((((0.14942924536134217) * (v1)) + ((0.14942924536134222) * (v2))) + ((0.14942924536134217) * (v3))) + (
    #             (0.9659258262890683) * (v4))) >= 3.465]
    # Eq: [(((((0.593146935671416) * (v1)) + ((0.4477599791339653) * (v2))) + ((0.3096198095942333) * (v3))) + (
    #             (0.593146935671416) * (v4))) < 5.985000000000002]
    # Eq: [v1 >= 0]
    # Eq: [v2 >= 0]
    # Eq: [v3 >= 0]
    # Eq: [v4 >= 0]


def main():
    test5()


if __name__ == "__main__":
    main()
