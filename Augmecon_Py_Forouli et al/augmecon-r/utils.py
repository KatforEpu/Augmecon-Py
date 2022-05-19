import copy
import random
import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib import cm
from pathlib import Path
import reader
import augmecon
from collections import Counter
import pandas as pd
# import viz
# import numpy as np


class Options:
    def __init__(
            self,
            filename,
            save_graph_output,
            save_excel_output,
            mc_iterations,
            mc_std_deviation_percentage,
            name_graph_axes,
            scale_graph_axes,
            graph_title,
            graph_color,
            nadir_undercut,
            plot_all_output_solutions,
            plot_cutoff,
            fixed_nadirs=None
    ):
        self.filename = filename
        self.save_graph_output = save_graph_output
        self.save_excel_output = save_excel_output
        self.mc_iterations = mc_iterations
        self.mc_std_deviation_percentage = mc_std_deviation_percentage
        self.name_graph_axes = name_graph_axes
        self.scale_graph_axes = scale_graph_axes
        self.graph_title = graph_title
        self.graph_color = graph_color
        self.nadir_undercut = nadir_undercut
        self.plot_all_output_solutions = plot_all_output_solutions
        self.plot_cutoff = plot_cutoff
        self.fixed_nadirs = fixed_nadirs


def flat_map(f, xs):
    ys = []
    for x in xs:
        ys.extend(f(x))
    return ys


def print_counter_vertically(ct):
    keys = list(ct)
    keys.sort(key=lambda y: ct[y], reverse=True)
    for x in keys:
        print(ct[x], x)


def random_color():
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    return color


def draw_pareto(moip, options: Options, solution_points=[], iterations=1):
    # viz.draw_multidimensional(moip)
    # return

    r = moip.undo_auto_swap_proxy
    solutions_counter = Counter(solution_points)
    is_monte_carlo = len(solutions_counter) > 0 and iterations > 1
    moip_solution_points = moip.get_solution_points()
    moip_solution_values = moip.get_solution_values()
    num_solutions = len(moip_solution_values)
    objectives_count = len(moip_solution_values[0])
    objective_values = [[moip_solution_values[i][x] for i in range(num_solutions)] for x in range(objectives_count)]
    data = [sheet for sheet in moip.original_model.Datasheets if 'objective' in sheet]

    if is_monte_carlo:
        temp_moip_solution_points = []
        objective_values = []
        sol_idx = 0
        for x in moip_solution_points:
            if solutions_counter[x] / options.mc_iterations > options.plot_cutoff:
                temp_moip_solution_points.append(x)
                objective_values.append(moip_solution_values[sol_idx])
            sol_idx += 1
        moip_solution_points = temp_moip_solution_points
        objective_values = [[objective_values[o][k] for o in range(len(objective_values))] for k in range(objectives_count)]

    distinct_sols = set(solution_points)
    distinct_sols = [x for x in distinct_sols if solutions_counter[x]/options.mc_iterations > options.plot_cutoff]

    opacity = .75
    if is_monte_carlo:
        if options.plot_all_output_solutions:
            opacity = [.75 if moip_solution_points.__contains__(sol) else .15 for sol in distinct_sols]

    marker_size = None
    if is_monte_carlo:
        if options.plot_all_output_solutions:
            marker_size = [1000 ** (solutions_counter[point] / iterations) for point in distinct_sols]
            objective_values = []
            for sheet in data:
                obj_vals_for_cur_objective = []
                for sol_point in distinct_sols:
                    solution_vector = sol_point.split(',')
                    current_obj_value = 0
                    for decision_variable in solution_vector:
                        name = moip.get_decision_variable_name_by_index(int(decision_variable))
                        tech = name.split('-')[0]
                        level = name.split('-')[1]
                        current_obj_value += moip.original_model.Datasheets[sheet][level][tech]
                    obj_vals_for_cur_objective.append(current_obj_value)
                objective_values.append(obj_vals_for_cur_objective)
        else:
            marker_size = [1000**(solutions_counter[point]/iterations) for point in moip_solution_points]

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(options.graph_title or moip.name, fontweight='bold', fontsize=14)

    if len(options.scale_graph_axes) < objectives_count:
        for x in range(len(options.scale_graph_axes), objectives_count):
            options.scale_graph_axes.append(1)

    if len(options.name_graph_axes) < objectives_count:
        for x in range(len(options.name_graph_axes), objectives_count):
            options.name_graph_axes.append('')

    if objectives_count < 2:
        raise Exception('Graph error: Less than 2 objectives in the model!')
    elif objectives_count == 2:
        ax = fig.add_subplot(6, 5, (1, 20)) if is_monte_carlo else fig.add_subplot()
        # ax.scatter(*objective_values, marker='o', s=marker_size)
        ax.scatter(
            [x * options.scale_graph_axes[r[0]] for x in objective_values[r[0]]],
            [y * options.scale_graph_axes[r[1]] for y in objective_values[r[1]]],
            marker='o', s=marker_size, alpha=opacity,
            c=options.graph_color or None)
        ax.set_xlabel(options.name_graph_axes[r[0]] or data[r[0]])
        ax.set_ylabel(options.name_graph_axes[r[1]] or data[r[1]])
        ax.margins(.1, .1)
        if is_monte_carlo:
            ms = fig.add_subplot(6, 5, (26, 27))
            points = distinct_sols if options.plot_all_output_solutions else moip_solution_points
            ms.scatter([(solutions_counter[point]/iterations) for point in points],
                       [0 for point in points], marker='o',
                       s=marker_size, c=options.graph_color or None)
            ms.set_xlabel('Occurrence (as % of iterations)')
            ms.margins(.1, .05)
    else:
        if objectives_count > 3:
            print('WARNING : maximum 3 objectives can be depicted! If more they\'ll be ignored in this graph')
        ax = fig.add_subplot() if len(solutions_counter) == 0 else fig.add_subplot(6, 5, (1, 20))
        graph = ax.scatter(
            # [x * options.scale_graph_axes[0] for x in objective_values[0]],
            # [y * options.scale_graph_axes[1] for y in objective_values[1]],
            # c=[z * options.scale_graph_axes[2] for z in objective_values[2]],
            [x * options.scale_graph_axes[r[0]] for x in objective_values[r[0]]],
            [y * options.scale_graph_axes[r[1]] for y in objective_values[r[1]]],
            c=[z * options.scale_graph_axes[r[2]] for z in objective_values[r[2]]],
            marker='o', cmap='gnuplot', s=marker_size, alpha=opacity)
        color_bar = plt.colorbar(graph)
        ax.set_xlabel(options.name_graph_axes[r[0]] or data[r[0]])
        ax.set_ylabel(options.name_graph_axes[r[1]] or data[r[1]])
        color_bar.set_label(options.name_graph_axes[r[2]] or data[r[2]])
        ax.margins(.1, .1)
        if is_monte_carlo:
            ms = fig.add_subplot(6, 5, (26, 27))
            points = distinct_sols if options.plot_all_output_solutions else moip_solution_points
            ms.scatter([(solutions_counter[point]/iterations) for point in points],
                       [0 for point in points], marker='o',
                       s=marker_size, c=options.graph_color or None)
            ms.set_xlabel('Occurrence (as % of iterations)')
            ms.margins(.1, .05)
    if options.save_graph_output:
        jpg = f"{Path().absolute()}\output\pareto_graph_{moip.name}.jpg"
        fig.savefig(jpg, format='jpg', dpi=800)

    # plt.show()


def run_model_from_excel(options: Options):
    problem = reader.read_excel_model(options.filename)
    moip = augmecon.MoipAugmeconR(problem,
                                  model_name=options.filename,
                                  min_to_nadir_undercut=options.nadir_undercut,
                                  fixed_nadirs=options.fixed_nadirs)
    r = moip.get_obj_ordered_by_ranges_desc()
    options = reorder_options(options, r)
    problem = reader.read_excel_model(options.filename, obj_order=r)
    moip = augmecon.MoipAugmeconR(problem,
                                  model_name=options.filename,
                                  min_to_nadir_undercut=options.nadir_undercut,
                                  fixed_nadirs=options.fixed_nadirs)
    moip.execute()
    moip.display_pareto_with_variable_names()
    draw_pareto(moip, options=options)
    print('total time of running the augmecon r algorithm', moip.after_aug_r - moip.before_aug_r)

    if options.save_excel_output:
        moip.save_solutions_to_excel()


def reorder_options(options: Options, r):
    if options.fixed_nadirs:
        options.fixed_nadirs = options.fixed_nadirs[2:]
    opts = copy.deepcopy(options)
    for j in range(len(r)):
        if opts.fixed_nadirs:
            opts.fixed_nadirs.append(None)
        opts.scale_graph_axes.append(1)
        opts.name_graph_axes.append('')
    if opts.fixed_nadirs:
        opts.fixed_nadirs = [(opts.fixed_nadirs[i - 1] or None) for i in r]
    opts.scale_graph_axes = [(opts.scale_graph_axes[i - 1] or 1) for i in r]
    opts.name_graph_axes = [(opts.name_graph_axes[i - 1] or '') for i in r]
    if opts.fixed_nadirs:
        opts.fixed_nadirs.insert(0, None)
        opts.fixed_nadirs.insert(0, None)
    return opts


def run_iterations(options: Options):
    if options.mc_iterations < 2:
        run_model_from_excel(options)
        return

    solution_points = []
    solution_values = []
    moip = {}

    problem_initial = reader.read_excel_model(options.filename)
    moip_initial = augmecon.MoipAugmeconR(problem_initial,
                                          model_name=options.filename,
                                          min_to_nadir_undercut=options.nadir_undercut,
                                          fixed_nadirs=options.fixed_nadirs)
    r = moip_initial.get_obj_ordered_by_ranges_desc()
    reordered_options = reorder_options(options, r)
    problem_initial = reader.read_excel_model(reordered_options.filename, obj_order=r)
    moip_initial = augmecon.MoipAugmeconR(problem_initial,
                                          model_name=reordered_options.filename,
                                          min_to_nadir_undercut=reordered_options.nadir_undercut,
                                          fixed_nadirs=reordered_options.fixed_nadirs)
    moip_initial.execute()
    initial_moip_solution_points = moip_initial.get_solution_points()
    initial_moip_solution_values = moip_initial.get_solution_values()

    for iteration in range(options.mc_iterations):
        problem = reader.read_excel_model(reordered_options.filename,
                                          reordered_options.mc_std_deviation_percentage, obj_order=r)
        moip = augmecon.MoipAugmeconR(problem,
                                      model_name=reordered_options.filename,
                                      min_to_nadir_undercut=reordered_options.nadir_undercut,
                                      fixed_nadirs=reordered_options.fixed_nadirs)
        moip.execute()
        solution_points += moip.get_solution_points()
        solution_values += moip.get_solution_values()

    total_portfolios = len(solution_points)
    total_unique_portfolios = len(set(solution_points))

    print('')
    print('')
    print('')
    print('======================================')
    print('             R E S U L T S           ')
    print('======================================')
    print('iterations:', options.mc_iterations)
    print('')
    print('total number of solutions:', total_portfolios)
    print('total number of UNIQUE solutions:', total_unique_portfolios)
    print('')
    print('------------------------------------')
    print('portfolios and times of occurrence:')
    print('------------------------------------')
    sols_in_text = map(
        lambda p: ';'.join(map(lambda t: moip.get_decision_variable_name_by_index(int(t)), p.split(','))),
        solution_points
    )
    counter_of_sols = Counter(sols_in_text)
    print_counter_vertically(counter_of_sols)
    print('------------------------------------')

    techs = flat_map(lambda z: z.split(','), solution_points)
    total_techs = len(techs)
    total_unique_techs = len(set(techs))
    print('')
    print('total num of techs:', total_techs)
    print('total num of UNIQUE techs:', total_unique_techs)
    print('')
    print('------------------------------------')
    print('techs and times of occurrence:')
    print('------------------------------------')
    techs_in_text = map(lambda index: moip.get_decision_variable_name_by_index(index), map(int, techs))
    counter_of_techs = Counter(techs_in_text)
    print_counter_vertically(counter_of_techs)
    print('------------------------------------')

    # -----------------------
    #  Save to excel file
    # -----------------------
    if options.save_excel_output:
        output_file = f"{Path().absolute()}\output\out_MC_{moip.name}.xlsx"
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

        # write sheet one with portfolios
        save_mc_solutions_to_excel(writer, moip_initial, initial_moip_solution_points,
                                   solution_points, solution_values)

        # write sheet two with techs
        columns = ['Times of Occurrence', 'Tech']
        sheet_rows = []
        keys = list(counter_of_techs)
        keys.sort(key=lambda y: counter_of_techs[y], reverse=True)
        for x in keys:
            sheet_rows.append([counter_of_techs[x], x])

        df2 = pd.DataFrame(sheet_rows, columns=columns)
        df2.to_excel(writer, sheet_name='techs', index=False)

        writer.save()

    # ----------------------------------------------
    #  Draw pareto front with sensitivity
    # ----------------------------------------------
    draw_pareto(moip_initial,
                options=reordered_options,
                solution_points=solution_points,
                iterations=reordered_options.mc_iterations)


def save_mc_solutions_to_excel(writer, moip_initial, initial_moip_solution_points,
                               solution_points, solution_values):
    # zip solution points and objective values, for easier access
    sol_idx = 0
    idx_point_values_arr = []
    for sol_point in solution_points:
        idx_point_values_arr.append([sol_idx, sol_point, solution_values[sol_idx]])
        sol_idx += 1

    column_headers = []
    rows = []

    data = moip_initial.original_model.Datasheets

    distinct_sols = set(solution_points)
    counter_of_sols = Counter(solution_points)

    for sol_point in distinct_sols:
        column_i = 0
        row = []
        solution_vector = sol_point.split(',')

        # -------------------------------------------------
        # uncomment section below to add 'objective value averages' columns (avg calculated among occurrences of a sol)
        # -------------------------------------------------
        # # calculate average objective values, among occurrences of solution
        # related_ipv = filter(lambda y: (sol_point == y[1]), idx_point_values_arr)
        # related_obj_vals = list(map(lambda y: y[2], related_ipv))
        # avg_obj_vals = np.average(related_obj_vals, axis=0)
        # # sheet headers for average objective values
        # for sheet in data:
        #     if 'objective' in sheet:
        #         column_i += 1
        #         if len(column_headers) < column_i:
        #             column_headers.append(sheet + ' (avg)')
        # # sheet data for average objective values
        # row += [i for i in avg_obj_vals]
        # -------------------------------------------------

        obj_vals = []
        for sheet in data:
            if 'objective' in sheet:
                # sheet headers for initial problem's objective values
                column_i += 1
                if len(column_headers) < column_i:
                    column_headers.append(sheet)
                # sheet data for initial problem's objective values
                current_obj_value = 0
                for decision_variable in solution_vector:
                    name = moip_initial.get_decision_variable_name_by_index(int(decision_variable))
                    tech = name.split('-')[0]
                    level = name.split('-')[1]
                    current_obj_value += data[sheet][level][tech]
                obj_vals.append(current_obj_value)
        row += [i for i in obj_vals]

        active_vars = ''
        # for each tech-level
        for decision_variable in solution_vector:
            name = moip_initial.get_decision_variable_name_by_index(int(decision_variable))
            tech = name.split('-')[0]
            level = name.split('-')[1]
            active_vars += name + ', '

            # make a column name
            column_i += 1
            if len(column_headers) < column_i:
                column_headers.append('')

            #  make a value
            row.append(name)

            obj_cnt = 0
            for sheet in data:
                # make 2 column names (abs value and percentage)
                column_i += 2
                if len(column_headers) < column_i:
                    column_headers.append(f'{sheet}_abs')
                    column_headers.append(f'{sheet}_%')

                #  make 2 values (abs value and percentage)
                row.append(data[sheet][level][tech])
                if 'objective' in sheet:
                    row.append(round(data[sheet][level][tech] * 100 / obj_vals[obj_cnt], 0))
                    obj_cnt += 1
                if 'constraint' in sheet:
                    row.append(round(data[sheet][level][tech] * 100 / data[sheet].index.name, 0))

        # make a value for in initial problem
        row.insert(0, 1 if sol_point in initial_moip_solution_points else 0)
        # make a value for active variables
        row.insert(0, active_vars[:-2])  # trimming last 2 chars cause they are , and space
        # make a value for times of occurrence
        row.insert(0, int(counter_of_sols[sol_point]))

        rows.append(row)
        rows.sort(key=lambda y: y[0], reverse=True)

    # make a column name for in initial problem
    column_headers.insert(0, 'In Initial Problem')
    # make a column name for active variables
    column_headers.insert(0, 'active variables')
    # make a column name for times of occurrence
    column_headers.insert(0, 'Times of Occurrence')

    df = pd.DataFrame(rows, columns=column_headers)
    df.to_excel(writer, sheet_name='portfolios', index=False)
