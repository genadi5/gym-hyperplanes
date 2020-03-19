from gym_hyperplanes.simplex.simplex import Simplex


def main():
    objective = ('minimize', '4x_1 + 1x_2')
    constraints = ['3x_1 + 1x_2 = 3', '4x_1 + 3x_2 >= 6', '1x_1 + 2x_2 <= 4']
    lp_system = Simplex(num_vars=2, constraints=constraints, objective_function=objective)
    print(lp_system.solution)
    print(lp_system.optimize_val)


if __name__ == "__main__":
    main()
