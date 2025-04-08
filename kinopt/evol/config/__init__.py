from kinopt.evol.config.constants import _parse_arguments

parse_arguments =_parse_arguments
METHOD, lb, ub, loss_type, include_regularization, estimate_missing_kinases, scaling_method, split_point, segment_points =  parse_arguments()

# Define the dictionary mapping kinases to their respective add_psites
kinase_to_psites = {
 "CDK5": 1,
 "TTK": 7,
 "GSK3B": 4,
 "MAP2K4": 4,
 "MAP2K1": 2,
 "MAP2K3": 1,
 "CDK4": 2
}

time_series_columns = [f'x{i}' for i in range(1, 15)]  # columns x1 to x14