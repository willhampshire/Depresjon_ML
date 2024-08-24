from typing import Any
import numpy as np
import pandas as pd


def print_verbose(arg: Any):
    """
    Prints information about the variable:
    - TYPE: the type of the variable
    - SHAPE: the shape of the variable (if applicable)
    - VALUE: the value of the variable
    """

    print(f"--- print_verbose ---\n")
    prefix = "\n\t"
    try:
        if isinstance(arg, np.ndarray):
            print(f"{prefix}TYPE: {type(arg)}")
            print(f"{prefix}SHAPE: {arg.shape}")
            print(f"{prefix}VALUE:\n{arg}")
        elif isinstance(arg, pd.DataFrame):
            pd.set_option("display.max_columns", None)
            arg.info()
            print(f"{prefix}TYPE: {type(arg)}")
            print(f"{prefix}SHAPE: {arg.shape}")
            print(f"{prefix}HEAD:\n{arg.head(3)}")
            print(f"{prefix}VALUE:\n{arg}")
        elif isinstance(arg, pd.Series):
            print(f"{prefix}TYPE: {type(arg)}")
            print(f"{prefix}SHAPE: {arg.shape}")
            print(f"{prefix}VALUE:\n{arg}")
        elif isinstance(arg, str):
            print(f"\tVALUE: '{arg}'")
        elif isinstance(arg, dict):
            print(f"{prefix}KEYS: {arg.keys()}")
            print(f"{prefix}VALUE:\n{arg}")
        else:
            print(f"{prefix}TYPE: {type(arg)}")
            print(f"{prefix}SHAPE: {np.shape(arg)}")
            print(f"{prefix}VALUE: {arg}")

    except AttributeError as e:
        print(f"error: {e.args[0]}\nfor type: {type(arg)}")
        print(arg)


if __name__ == "__main__":
    test = {"key": 3458}

    print_verbose(test)
