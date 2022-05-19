import pyomo.environ as p
from pyomo.environ import *
import pandas as pd
from pathlib import Path
import numpy as np


def four_kp_model(filename):
    model = ConcreteModel()

    # Define input files
    xlsx = pd.ExcelFile(f"{Path().absolute()}/input/{filename}.xlsx")
    a = pd.read_excel(xlsx, index_col=0, sheet_name='a').to_numpy()
    b = pd.read_excel(xlsx, index_col=0, sheet_name='b').to_numpy()
    c = pd.read_excel(xlsx, index_col=0, sheet_name='c').to_numpy()

    # Define variables
    model.ITEMS = Set(initialize=range(len(a[0])))
    model.DecisionVariable = Var(model.ITEMS, within=p.Binary)

    # --------------------------------------
    #   Define the objective functions
    # --------------------------------------

    def objective1(model):
        return sum(c[0][i]*model.DecisionVariable[i] for i in model.ITEMS)

    def objective2(model):
        return sum(c[1][i]*model.DecisionVariable[i] for i in model.ITEMS)

    def objective3(model):
        return sum(c[2][i]*model.DecisionVariable[i] for i in model.ITEMS)

    def objective4(model):
        return sum(c[3][i]*model.DecisionVariable[i] for i in model.ITEMS)

    # --------------------------------------
    #   Define the regular constraints
    # --------------------------------------

    def constraint1(model):
        return sum(a[0][i]*model.DecisionVariable[i] for i in model.ITEMS) <= b[0][0]

    def constraint2(model):
        return sum(a[1][i]*model.DecisionVariable[i] for i in model.ITEMS) <= b[1][0]

    def constraint3(model):
        return sum(a[2][i]*model.DecisionVariable[i] for i in model.ITEMS) <= b[2][0]

    def constraint4(model):
        return sum(a[3][i]*model.DecisionVariable[i] for i in model.ITEMS) <= b[3][0]

    # --------------------------------------
    #   Add components to the model
    # --------------------------------------

    # Add the constraints to the model
    model.con1 = Constraint(rule=constraint1)
    model.con2 = Constraint(rule=constraint2)
    model.con3 = Constraint(rule=constraint3)
    model.con4 = Constraint(rule=constraint4)

    # Add the objective functions to the model using ObjectiveList(). Note
    # that the first index is 1 instead of 0!
    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=objective1(model), sense=maximize)
    model.obj_list.add(expr=objective2(model), sense=maximize)
    model.obj_list.add(expr=objective3(model), sense=maximize)
    model.obj_list.add(expr=objective4(model), sense=maximize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    return model


def get_non_white_str_list(enumerable):
    return list(x for x in enumerable if not str.startswith(x, 'Unnamed'))


def read_excel_model(xlsx_filename, randomness_percentage=[], obj_order=None):

    def randomize_value(avg, std_dev_percentage=0):
        is_negative = avg < 0
        if is_negative:
            avg = -avg
        num_reps = 1
        decimal_digits = 0
        std_dev = std_dev_percentage * avg
        random_values = np.random.normal(avg, std_dev, num_reps).round(decimal_digits)
        output = 0
        if is_negative:
            output = int(-random_values[0])
        else:
            output = int(random_values[0])
        return output

    def form_objective(mdl, data):
        objective_expr = 0
        idx = 0
        cols = get_non_white_str_list(data.keys())
        rows = get_non_white_str_list(data[cols[0]].keys())
        if len(randomness_percentage) < len(rows):
            for x in range(len(randomness_percentage), len(rows)):
                randomness_percentage.append(0)

        tech_index = 0
        for tech in rows:
            for level in cols:
                coefficient = randomize_value(data[level][tech], randomness_percentage[tech_index] or 0)
                objective_expr += coefficient * mdl.DecisionVariable[idx]
                idx += 1
            tech_index += 1

        return objective_expr

    def form_constraint(mdl, data):
        constraint_left_side = 0
        idx = 0
        cols = get_non_white_str_list(data.keys())
        rows = get_non_white_str_list(data[cols[0]].keys())

        for tech in rows:
            for level in cols:
                coefficient = data[level][tech]
                constraint_left_side += coefficient * mdl.DecisionVariable[idx]
                idx += 1

        return constraint_left_side <= data.index.name

    def form_mutual_exclusiveness_constraint(mdl, cols, rows, current_row):
        expression = 0
        idx = 0
        for tech in rows:
            for level in cols:
                if tech == current_row:
                    expression += mdl.DecisionVariable[idx]
                idx += 1
        return expression <= 1

    # --------------------------------------
    #   READ EXCEL FILE
    # --------------------------------------
    xlsx = pd.ExcelFile(f"{Path().absolute()}/input/{xlsx_filename}.xlsx")
    sheets = pd.read_excel(xlsx, sheet_name=None, index_col=0)

    # --------------------------------------
    #   VALIDATE AND GET DATAFRAME INDEXES
    # --------------------------------------
    _sheet_names = xlsx.sheet_names
    _row_names = get_non_white_str_list(sheets[xlsx.sheet_names[0]].index)
    _col_names = get_non_white_str_list(sheets[xlsx.sheet_names[0]].columns)
    for sheet_name in sheets:
        sheet_data = sheets[sheet_name]
        column_names = get_non_white_str_list(sheet_data.columns)
        row_names = get_non_white_str_list(sheet_data.index)
        if not _row_names == row_names:
            raise Exception(f'"{sheet_name}" row names differ from the ones in first sheet')
        if not _col_names == column_names:
            raise Exception(f'"{sheet_name}" column names differ from the ones in first sheet.')
    # print("-----------------------")
    # print(_sheet_names)
    # print(_row_names)
    # print(_col_names)
    # print("-----------------------")

    # --------------------------------------
    #   CREATE MODEL
    # --------------------------------------
    model = p.ConcreteModel()
    model.Datasheets = sheets
    model.Techs = p.Set(initialize=_row_names)
    model.Levels = p.Set(initialize=_col_names)
    model.ITEMS = p.Set(initialize=range(len(_row_names) * len(_col_names)))
    model.DecisionVariable = p.Var(model.ITEMS, within=p.Binary)
    model.obj_list = p.ObjectiveList()
    model.constraint_list = p.ConstraintList()
    model.undo_auto_swap_proxy = np.argsort(obj_order)

    # --------------------------------------
    #   ADD OBJECTIVES AND CONSTRAINTS
    # --------------------------------------
    cons = [sheet_name for sheet_name in sheets if 'constraint' in sheet_name]
    obj = [sheet_name for sheet_name in sheets if 'objective' in sheet_name]

    if obj_order:  # read objectives in specific order
        # need to rearrange sheet names for objectives
        obj = [obj[i-1] for i in obj_order]

        # need also to rearrange the keys/values in model.Datasheets
        temp = {}
        obj_idx = 0
        for sheet_name in sheets:
            if 'constraint' in sheet_name:
                temp[sheet_name] = sheets[sheet_name]
            if 'objective' in sheet_name:
                temp[obj[obj_idx]] = sheets[obj[obj_idx]]
                obj_idx += 1
        model.Datasheets = temp

    for sheet_name in cons:
        model.constraint_list.add(expr=form_constraint(model, sheets[sheet_name]))
    for sheet_name in obj:
        model.obj_list.add(expr=form_objective(model, sheets[sheet_name]), sense=p.maximize)

    for row in _row_names:
        model.constraint_list.add(expr=form_mutual_exclusiveness_constraint(model, _col_names, _row_names, row))

    model.obj_list.deactivate()

    # for x in model.obj_list:
    #     print('objective')
    #     print(model.obj_list[x].expr)
    # for x in model.constraint_list:
    #     print('constraint')
    #     print(model.constraint_list[x].expr)

    return model

