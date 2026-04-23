import json
import numpy as np

def convert_response2json(output, step_1=False, step_2=False):
    objs = []
    start_pos = 0  # starting position for searching

    while start_pos < len(output):
        # Find the next '{' character
        start_pos = output.find('{', start_pos)
        if start_pos == -1:
            break

        # From the current '{', try to decode substrings to find valid JSON
        for end_pos in range(len(output), start_pos, -1):
            try:
                obj = json.loads(output[start_pos:end_pos])
                if step_1:
                    # if all(key in obj for key in ['name', 'value', 'description']) and len(obj) == 3:
                    if all(key in obj for key in ['description']) and len(obj) == 1:
                        objs.append(obj)
                elif step_2:
                    if all(key in obj for key in ['name', 'value', "type"]) and len(obj) == 3:
                        objs.append(obj)
                else:
                    objs.append(obj)
                start_pos = end_pos  # move past this valid object
                break
            except json.JSONDecodeError:
                pass  # Continue with a shorter substring

        start_pos += 1  # Move to the next character after current '{'

    return objs

def convert_to_years(value):
    """
    Convert age values to years if they are in weeks, months, or days.
    Returns the converted value in years as an integer.
    """
    if isinstance(value, (int, float, np.integer)):
        return float(value)

    value = value.lower().strip()
    if "week" in value:
        return float(value.split()[0]) / 52  # Approximate weeks in a year
    elif "month" in value:
        return float(value.split()[0]) / 12  # Approximate months in a year
    elif "day" in value:
        return float(value.split()[0]) / 365  # Approximate days in a year
    elif "year" in value:
        return float(value.split()[0])
    else:
        return float(value)