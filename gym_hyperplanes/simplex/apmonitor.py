from APMonitor.apm import *


def main():
    # Solve optimization problem
    sol = apm_solve('test', 3)

    # Access solution
    x1 = sol['x1']
    x2 = sol['x2']

    print(sol)


if __name__ == "__main__":
    main()
