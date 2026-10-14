"""The tree family's frozen tuning budget: eight LightGBM settings, each
trained with early stopping on the validation split. Every tree model in
every set-up gets this same grid, so differences between set-ups are not
differences in tuning effort."""

GRID = (
    {"num_leaves": 64, "learning_rate": 0.05, "scale_pos_weight": 1},
    {"num_leaves": 64, "learning_rate": 0.05, "scale_pos_weight": 5},
    {"num_leaves": 64, "learning_rate": 0.05, "scale_pos_weight": 10},
    {"num_leaves": 256, "learning_rate": 0.05, "scale_pos_weight": 1},
    {"num_leaves": 256, "learning_rate": 0.05, "scale_pos_weight": 5},
    {"num_leaves": 256, "learning_rate": 0.05, "scale_pos_weight": 10},
    {"num_leaves": 256, "learning_rate": 0.02, "scale_pos_weight": 5},
    {"num_leaves": 1024, "learning_rate": 0.05, "scale_pos_weight": 5},
)
FIXED = {
    "objective": "binary",
    "metric": "average_precision",
    "lambda_l2": 1.0,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_threads": 10,
    "verbosity": -1,
    "max_bin": 63,
    "seed": 0,
}
NUM_ROUND = 500
EARLY_STOP = 30
