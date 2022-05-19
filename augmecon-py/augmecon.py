import datetime
import pyomo.environ as p
import numpy as np
import copy
import math
import pandas as pd
from pathlib import Path


class Solution:
    def __init__(self, point, objective_values):
        self.point = point
        self.objective_values = objective_values

    def display(self):
        print('values:', self.objective_values, 'point:', self.point)


class MoipAugmeconR:
    def __init__(
            self,
            pyomo_model,
            solver=p.SolverFactory('gurobi', solver_io='python'),
            model_name="Unknown",
            min_to_nadir_undercut=.8,
            fixed_nadirs=None
    ):

        self.original_model = pyomo_model
        self.undo_auto_swap_proxy = pyomo_model.undo_auto_swap_proxy
        self.model = copy.deepcopy(self.original_model)
        self.created_date = datetime.datetime.now()
        self.name = model_name + '_' + str(self.created_date).replace(':', '-')
        self.solver = solver
        self.fixed_nadirs = fixed_nadirs

        self.objectives_count = len(self.original_model.obj_list)
        self.decision_var_count = len(self.original_model.DecisionVariable)

        # payoff table props
        self.min_to_nadir_undercut = min_to_nadir_undercut
        self.payoff_table = np.full((self.objectives_count + 1, self.objectives_count + 1), np.inf)
        self.objective_maximums = np.zeros((self.objectives_count + 1))
        self.nadir_values = np.zeros((self.objectives_count + 1))

        # augmecon model props
        self.augmecon_model = copy.deepcopy(self.original_model)
        self.current_grid_point = np.zeros((self.objectives_count + 1))
        self.flags = {}
        self.models_solved = 0
        self.all_solutions = {}
        self.pareto_front = {}
        self.infeasibilities = 0

        # run augmecon
        self.before_aug_r = datetime.datetime.now()
        self.after_aug_r = datetime.datetime.now()
        # self.execute()

    def get_obj_ordered_by_ranges_desc(self):
        self.create_payoff_table()
        ranges = [self.objective_maximums[i] - self.nadir_values[i] for i in self.range_objectives()]
        indices = np.argsort(ranges)
        r = list(map(lambda x: x + 1, indices[::-1]))
        return r

    def execute(self):
        self.create_payoff_table()
        self.print_payoff_table()
        self.build_augmecon_problem()
        # self.print_augmecon_problem()
        self.before_aug_r = datetime.datetime.now()
        self.run_augmecon_r()
        self.after_aug_r = datetime.datetime.now()
        self.display_pareto()
        print('total time', datetime.datetime.now() - self.created_date)
        print('total time of running the augmecon r algorithm', self.after_aug_r - self.before_aug_r)
        print('')
        print("total models calculated", self.models_solved)
        print("infeasibilities", self.infeasibilities)
        print("pareto sol num", len(self.pareto_front.values()))

    def range_objectives(self):
        return range(1, self.objectives_count + 1)

    def range_slacks(self):
        return range(2, self.objectives_count + 1)

    def range_decision_vars(self):
        return range(self.decision_var_count)

    def reset_solution_counters(self):
        self.models_solved = 0
        self.infeasibilities = 0

    def solve_model(self):
        self.solver.options['mipgap'] = 0.0
        return self.solver.solve(self.model)

    def solve_augmecon_model(self):
        self.models_solved += 1
        self.solver.options['mipgap'] = 0.0
        return self.solver.solve(self.augmecon_model)

    def activate_objective(self, i):
        self.model.obj_list[i].activate()

    def deactivate_objective(self, i):
        self.model.obj_list[i].deactivate()

    def deactivate_all_objectives(self):
        self.model.obj_list.deactivate()

    def create_payoff_table(self):
        for i in self.range_objectives():
            self.activate_objective(i)
            self.solve_model()
            self.payoff_table[i, i] = self.model.obj_list[i]()
            self.deactivate_objective(i)
            self.objective_maximums[i] = self.model.obj_list[i]()

        self.model.aux_con = p.ConstraintList()
        aux_con_index = 0
        for i in self.range_objectives():
            self.model.aux_con.add(expr=self.model.obj_list[i].expr == self.payoff_table[i, i])
            aux_con_index += 1
            for j_plus in range(i, i + self.objectives_count - 1):
                j = (j_plus % self.objectives_count) + 1
                if i != j:
                    self.activate_objective(j)
                    self.solve_model()
                    temp_value = self.model.obj_list[j]()
                    self.deactivate_objective(j)
                    self.payoff_table[i, j] = round(temp_value, 10)
                    self.model.aux_con.add(expr=self.model.obj_list[j].expr == temp_value)
                    aux_con_index += 1
            for x in range(1, aux_con_index + 1):
                self.model.aux_con[x].deactivate()
        del self.model.aux_con

        for j in self.range_objectives():
            self.nadir_values[j] = min(
                round(min(self.payoff_table[i, j] for i in self.range_objectives()) * self.min_to_nadir_undercut, 0),
                round(min(self.payoff_table[i, j] for i in self.range_objectives()) * (1/self.min_to_nadir_undercut), 0)
            )

    def print_payoff_table(self):
        print('')
        print('============================')
        print('        payoff table        ')
        print('============================')
        print('')
        print(self.payoff_table)
        print('')
        print('=======      max     =======\n', self.objective_maximums)
        print('')
        print('=======     nadir    =======\n', self.nadir_values)
        print('( note: undercut factor', self.min_to_nadir_undercut, ')')
        print('')
        if self.fixed_nadirs:
            print('payoff nadirs will be ignored, due to user setting exact nadir values for the grid.')
            print(self.fixed_nadirs)
        print('')
        print('')

    def build_augmecon_problem(self):
        epsilon = 10e-3
        slack_epsilon = [10 ** (2-x) for x in range(self.objectives_count+1)]

        self.augmecon_model.SlackVarSet = p.Set(initialize=[x for x in self.range_slacks()])
        self.augmecon_model.SlackVar = p.Var(self.augmecon_model.SlackVarSet, within=p.NonNegativeReals)
        self.augmecon_model.ObjRHS = p.Param(self.augmecon_model.SlackVarSet, within=p.Reals, mutable=True)

        for i in self.range_slacks():
            self.augmecon_model.obj_list[1].expr = \
                self.augmecon_model.obj_list[1].expr \
                + epsilon * slack_epsilon[i] * \
                self.augmecon_model.SlackVar[i] / (self.objective_maximums[i] - self.nadir_values[i])

        self.augmecon_model.augmecon_objective = \
            p.Objective(expr=self.augmecon_model.obj_list[1].expr, sense=p.maximize)

        self.augmecon_model.con_list = p.ConstraintList()
        for i in self.range_slacks():
            if self.augmecon_model.obj_list[i].sense == p.minimize:
                self.augmecon_model.con_list.add(
                    expr=self.augmecon_model.obj_list[i].expr
                    + self.augmecon_model.SlackVar[i] == self.augmecon_model.ObjRHS[i])
            if self.augmecon_model.obj_list[i].sense == p.maximize:
                self.augmecon_model.con_list.add(
                    expr=self.augmecon_model.obj_list[i].expr
                    - self.augmecon_model.SlackVar[i] == self.augmecon_model.ObjRHS[i])

        # self.augmecon_model.obj_list.deactivate()

    def print_augmecon_problem(self):
        print('============================')
        print('     Augmecon R problem     ')
        print('============================')
        print('')
        print('== New Objective function ==')
        print(self.augmecon_model.augmecon_objective.expr)
        print('')
        print('===== New Constraints ======')
        for i in self.range_slacks():
            print(self.augmecon_model.con_list[i-1].expr)
        print('')
        print('')

    def grid_point_key(self):
        separator = ','
        return separator.join(map(str, map(int, self.current_grid_point)))

    @staticmethod
    def flag_point_key(flags_point):
        separator = ','
        return separator.join(map(str, map(int, flags_point)))

    def is_flagged_for_skipping(self, key):
        return self.flags.__contains__(key) and self.flags[key] > 0

    @staticmethod
    def is_solution_status_ok(solution):
        return solution.solver.status == p.SolverStatus.ok or \
               solution.solver.status == p.SolverStatus.warning

    @staticmethod
    def is_solution_optimal(solution):
        return solution.solver.termination_condition == p.TerminationCondition.optimal

    @staticmethod
    def is_solution_infeasible(solution):
        return solution.solver.termination_condition == p.TerminationCondition.infeasibleOrUnbounded or \
               solution.solver.termination_condition == p.TerminationCondition.infeasible

    def note_solution(self):
        self.all_solutions[str(self.models_solved)] = \
            Solution(
                [round(p.value(self.augmecon_model.DecisionVariable[i]), 1) for i in self.range_decision_vars()],
                [round(self.augmecon_model.obj_list[i](), 1) for i in self.range_objectives()]
            )
        print("--> found SOLUTION",
              self.models_solved,
              [round(self.augmecon_model.obj_list[i](), 1) for i in self.range_objectives()],
              "***",
              [round(self.augmecon_model.SlackVar[i](), 1) for i in self.range_slacks()],
              )

    def flag_dominated_points(self):
        current_flags_point = np.zeros(self.objectives_count+1)
        self.nested_loops_flag_dominated(self.objectives_count, current_flags_point)

    def nested_loops_flag_dominated(self, obj_idx, current_flags_point):
        if obj_idx > 2:
            fromm = int(self.current_grid_point[obj_idx])
            to = int(self.current_grid_point[obj_idx] + self.augmecon_model.SlackVar[obj_idx]())
            for point in range(fromm, to+1):
                current_flags_point[obj_idx] = point
                self.nested_loops_flag_dominated(obj_idx - 1, current_flags_point)
        else:  # obj_idx == 2:
            current_flags_point[obj_idx] = self.current_grid_point[obj_idx]
            key = self.flag_point_key(current_flags_point)
            self.flags[key] = math.floor(self.augmecon_model.SlackVar[obj_idx]() + 1)

    def flag_infeasibility_points(self):
        current_flags_point = np.zeros(self.objectives_count+1)
        self.nested_loops_flag_infeasibility(self.objectives_count, current_flags_point)

    def nested_loops_flag_infeasibility(self, obj_idx, current_flags_point):
        if obj_idx > 2:
            fromm = int(self.current_grid_point[obj_idx])
            to = int(self.objective_maximums[obj_idx])
            for point in range(fromm, to + 1):
                current_flags_point[obj_idx] = point
                self.nested_loops_flag_infeasibility(obj_idx - 1, current_flags_point)
        else:  # obj_idx == 2:
            current_flags_point[obj_idx] = self.current_grid_point[obj_idx]
            key = self.flag_point_key(current_flags_point)
            self.flags[key] = round(self.objective_maximums[obj_idx] + 1, 0)

    def loop_through_grid(self, objective_idx):
        max_value = self.objective_maximums[objective_idx]
        cur = self.nadir_values[objective_idx]
        if self.fixed_nadirs:
            cur = self.fixed_nadirs[objective_idx]
        while cur <= max_value:
            self.current_grid_point[objective_idx] = round(cur, 0)
            self.augmecon_model.ObjRHS[objective_idx] = round(cur, 0)
            jump = self.calculate_jump(objective_idx)
            cur += jump

    def calculate_jump(self, objective_idx):
        if objective_idx > 2:
            self.loop_through_grid(objective_idx - 1)
            return 1

        key = self.grid_point_key()

        if self.is_flagged_for_skipping(key):
            return int(self.flags[key])

        sol = self.solve_augmecon_model()

        if not self.is_solution_status_ok(sol):
            print("ERROR solver status \n", sol)
            raise Exception("solver error")

        elif self.is_solution_optimal(sol):
            self.note_solution()
            self.flag_dominated_points()
            if self.augmecon_model.SlackVar[objective_idx]() > 0:
                return math.floor(self.augmecon_model.SlackVar[objective_idx]() + 1)
            else:
                return 1

        elif self.is_solution_infeasible(sol):
            print("infeasibility")
            self.infeasibilities += 1
            self.flag_infeasibility_points()
            return round(self.objective_maximums[objective_idx] + 1, 0)

        else:
            raise Exception("case of unhandled solution")

    def run_augmecon_r(self):
        self.reset_solution_counters()
        self.loop_through_grid(self.objectives_count)

    def display_pareto(self):
        for x in self.all_solutions.keys():
            unique = True
            for y in self.pareto_front.keys():
                comparison = np.array(self.all_solutions[x].point) == np.array(self.pareto_front[y].point)
                unique = not comparison.all()
                if not unique:
                    break
            if unique:
                self.pareto_front[x] = self.all_solutions[x]

        print('')
        print('============================')
        print('        pareto front        ')
        print('============================')
        [print('values:', x.objective_values, 'point:', x.point) for x in self.pareto_front.values()]
        print('')

    def get_solution_points(self):
        solution_keys = []
        for solution_idx in self.pareto_front:
            solution_vector = self.pareto_front[solution_idx].point
            key_vector = \
                [decision_variable
                 for decision_variable in range(len(solution_vector)) if solution_vector[decision_variable] > 0]
            separator = ','
            key = separator.join(map(str, key_vector))
            solution_keys.append(key)
        return solution_keys

    def get_solution_values(self):
        sol_vals = []
        for solution_idx in self.pareto_front:
            sol_vals.append(self.pareto_front[solution_idx].objective_values)
        return sol_vals

    def display_pareto_with_variable_names(self):
        print('')
        print('')
        for solution_idx in self.pareto_front:
            solution_vector = self.pareto_front[solution_idx].point
            values_vector = self.pareto_front[solution_idx].objective_values
            print('')
            print('===========================')
            print('solution', solution_idx, values_vector)
            print('===========================')
            for decision_variable in range(len(solution_vector)):
                if solution_vector[decision_variable] > 0:
                    name = self.get_decision_variable_name_by_index(decision_variable)
                    tech = name.split('-')[0]
                    level = name.split('-')[1]
                    data = self.original_model.Datasheets
                    contribution = []
                    obj_cnt = 0
                    for sheet in data:
                        contribution.append(sheet)
                        contribution.append(data[sheet][level][tech])
                        if 'objective' in sheet:
                            contribution.append(round(data[sheet][level][tech] * 100 / values_vector[obj_cnt], 0))
                            obj_cnt += 1
                        if 'constraint' in sheet:
                            contribution.append(round(data[sheet][level][tech] * 100 / data[sheet].index.name, 0))
                    print(name, contribution)
            print('===========================')

    def get_decision_variable_name_by_index(self, dv):
        tech_names = self.model.Techs
        level_names = self.model.Levels
        num_of_levels = len(level_names)
        return \
            tech_names[math.ceil((dv + 1) / num_of_levels)] + \
            '-' + \
            level_names[(dv + 1 - (math.ceil((dv + 1) / num_of_levels) - 1) * num_of_levels)]

    def save_solutions_to_excel(self):
        indexes = []
        columns = []
        output_data = []

        data = self.original_model.Datasheets

        for solution_idx in self.pareto_front:
            column_i = 0
            # out_solution = []
            solution_vector = self.pareto_front[solution_idx].point
            values_vector = self.pareto_front[solution_idx].objective_values

            # the index for each solution
            indexes.append(f'solution {solution_idx}')

            # make a column name for each objective
            for sheet in data:
                if 'objective' in sheet:
                    column_i += 1
                    if len(columns) < column_i:
                        columns.append(sheet)

            # make a value for each objective
            out_solution = values_vector

            # for each tech-level
            for decision_variable in range(len(solution_vector)):
                # if it is selected
                if solution_vector[decision_variable] > 0:
                    name = self.get_decision_variable_name_by_index(decision_variable)
                    tech = name.split('-')[0]
                    level = name.split('-')[1]

                    # make a column name
                    column_i += 1
                    if len(columns) < column_i:
                        columns.append('')

                    #  make a value
                    out_solution.append(name)

                    obj_cnt = 0
                    for sheet in data:
                        # make 2 column names (abs value and percentage)
                        column_i += 2
                        if len(columns) < column_i:
                            columns.append(f'{sheet}_abs')
                            columns.append(f'{sheet}_%')

                        #  make 2 values (abs value and percentage)
                        out_solution.append(data[sheet][level][tech])
                        if 'objective' in sheet:
                            out_solution.append(round(data[sheet][level][tech] * 100 / values_vector[obj_cnt], 0))
                            obj_cnt += 1
                        if 'constraint' in sheet:
                            out_solution.append(round(data[sheet][level][tech] * 100 / data[sheet].index.name, 0))

            output_data.append(out_solution)

        output_file = f"{Path().absolute()}\output\out_{self.name}.xlsx"
        writer = pd.ExcelWriter(output_file, engine='xlsxwriter')

        df = pd.DataFrame(output_data,
                          index=indexes,
                          columns=columns)
        df.to_excel(writer, sheet_name='pareto_solutions', index=True)

        writer.save()
