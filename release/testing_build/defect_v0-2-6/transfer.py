import torch

def prediction_transfer(model_outputs):
    """
    Transfer model prediction to normalized scores and rearrange them.
    Args:
        model_outputs: the output of the model, which is a vector with 7
        elements. Note that the saturation is the last element in this vector.
    Returns:
        The final scores of all 7 defects in a vector.
    """
    trans_outputs = model_outputs[:-1]  # take the first 6 defects
    sat_output = model_outputs[-1]      # take the last defect (saturation)

    x_value = [x for x in range(6)]
    y_value = [0., 0.05, 0.15, 0.3, 0.55, 1.0]
    linear_inter = lambda x_0, y_0, x_1, y_1, x: y_0 + (y_1 - y_0) / (x_1 - x_0) * (x - x_0)
    value_trans = []
    for value in trans_outputs:
        if value <= 1:
            value = linear_inter(x_value[0], y_value[0], x_value[1], y_value[1], value)
        elif 1 < value <= 2:
            value = linear_inter(x_value[1], y_value[1], x_value[2], y_value[2], value)
        elif 2 < value <= 3:
            value = linear_inter(x_value[2], y_value[2], x_value[3], y_value[3], value)
        elif 3 < value <= 4:
            value = linear_inter(x_value[3], y_value[3], x_value[4], y_value[4], value)
        else:
            value = linear_inter(x_value[4], y_value[4], x_value[5], y_value[5], value)
        value_trans.append(value)

     # rearrange to match with the predefined order
    return value_trans[:2] + [sat_output] + value_trans[2:]
