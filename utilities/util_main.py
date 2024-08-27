import ast
from transformers import pipeline
import os
import scipy.io.wavfile

def transform_to_list(output: str):
    """Transforms a string representation of a list into an actual Python list."""
    try:
        # Convert the string into a Python list using ast.literal_eval
        output_list = ast.literal_eval(output)
        return output_list
    except (ValueError, SyntaxError) as e:
        print(f"Error converting string to list: {e}")
        return []
    