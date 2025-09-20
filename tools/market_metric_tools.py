from .utils import get_generated_functions_excluding_called

def get_metricCoder_generated_functions():
    """
    Get all dynamically generated functions in the current module, excluding functions that are called by other functions.

    Returns:
        list: Returns a list of all dynamically generated function objects that are not called by other functions.
    """
    import sys
    import traceback

    try:
        # Get the current module object
        current_module = sys.modules[__name__]

        # Use the common utility function to get functions excluding called ones
        return get_generated_functions_excluding_called(current_module)

    except Exception as e:
        print(f"Error in get_metricCoder_generated_functions: {e}")
        print(traceback.format_exc())
        return []
