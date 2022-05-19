from utils import *

input_files = [
    'sample_input', ]

for file in input_files:
    opts = Options(
        filename=file,
        save_graph_output=True,
        save_excel_output=True,
        mc_iterations=5,  # > 1 for monte carlo
        mc_std_deviation_percentage=[.05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, .05, ],
        # ^^^ .2 corresponds to std deviation equal to 20% of the mean value. only valid for iterations > 1.
        # ^^^ percentage is set by decision variable. order of decision variables same as order of lines in excel sheet
        name_graph_axes=['1st objective', '2nd objective', '3rd objective'],
        # ^^^ order of names is linked to the order of objectives in the input sheet.
        # ^^^ First name goes to 1st objective etc
        scale_graph_axes=[1, 1, 1],
        # ^^^ order of scales is linked to the order of objectives in the input sheet.
        # ^^^ First scale goes to 1st objective etc
        graph_title=f'{file}',
        graph_color='blue',
        nadir_undercut=.7,
        plot_all_output_solutions=False,
        # ^^^ for monte carlo iterations. 'False' plots only initial problem's solutions.
        # ^^^ 'True' also prints the rest with opacity
        plot_cutoff=0.00,
        # ^^^ 0.00 to 1.00: threshold for a solution's robustness, so that the solution appears in the plot.
    )
    run_iterations(opts)
