def find_closest_point(point, required_class, hp_states):
    classifier = DeepHyperplanesClassifier(hp_states)
    y = classifier.predict(np.array([point]), required_class)
    if y[0] is not None:  # we found area for point
        if y[0] == required_class:  # this is our class!!!
            print('GREAT!!!!!!')
            return True, point, None

    results = []
    constraints_set_list = []
    for hp_state in hp_states:
        # REMOVE AFTER DEBUG
        powers = np.array([pow(2, i) for i in range(len(hp_state.hp_dist))])
        # REMOVE AFTER DEBUG
        constraints_sets = hp_state.get_class_constraint(required_class)
        if len(constraints_sets) > 0:
            features_minimums = hp_state.get_features_minimums()
            features_maximums = hp_state.get_features_maximums()

            for i, constraints_set in enumerate(constraints_sets):
                sys.stdout = open(os.devnull, "w")
                try:
                    m = GEKKO(remote=False)  # Initialize gekko
                    vars = generate_vars_objective(m, features_minimums, features_maximums, point)
                    generate_constraints(m, vars, constraints_set.get_constraints())
                    m.solve(disp=False)  # Solve
                    closest_point = [var.value[0] for var in vars]

                    results.append((closest_point, m.options.objfcnval))
                    constraints_set_list.append(constraints_set)

                    # REMOVE AFTER DEBUG
                    array = np.dot(np.array(closest_point), hp_state.hp_state) - hp_state.hp_dist
                    area = np.bitwise_or.reduce(powers[array > 0])

                    new_point = [(1 + 0.01) * t - 0.01 * f for f, t in zip(point, closest_point)]
                    new_array = np.dot(np.array(new_point), hp_state.hp_state) - hp_state.hp_dist
                    new_area = np.bitwise_or.reduce(powers[new_array > 0])

                    predict_state = HyperplanesClassifier(hp_state).predict(np.array([closest_point]))
                    if predict_state[0] != required_class:
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    new_predict_state = HyperplanesClassifier(hp_state).predict(np.array([new_point]))
                    if new_predict_state[0] != required_class:
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    predict_states = DeepHyperplanesClassifier(hp_states).predict(np.array([closest_point]))
                    if predict_states[0] != required_class:
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    new_predict_states = DeepHyperplanesClassifier(hp_states).predict(np.array([new_point]))
                    if new_predict_states[0] != required_class:
                        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    # REMOVE AFTER DEBUG
                except:
                    pass
                sys.stdout = sys.__stdout__

            # print("Result: " + str(results))
    min_distance = 0
    the_closest_point = None
    the_closest_constraints_set = None
    for result, constraints_set in zip(results, constraints_set_list):
        if the_closest_point is None:
            the_closest_point = result
            min_distance = math.sqrt(sum(map(lambda x: x * x, result[0])))
            the_closest_constraints_set = constraints_set
        else:
            distance = 0
            # their squares
            for s, d in zip(point, result[0]):
                distance += pow(d - s, 2)
            distance = math.sqrt(distance)

            if distance < min_distance:
                min_distance = distance
                the_closest_point = result
                the_closest_constraints_set = constraints_set
    print(the_closest_point)
    return False, the_closest_point[0], the_closest_constraints_set
