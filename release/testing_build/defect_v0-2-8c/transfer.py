import torch
import torch.nn.functional as F

OUTPUT_NUM = 7  # the final output number
SAT_IDX = 2     # the index of saturation
HEAD_NUM = 3    # the number of prediction heads

def convert(raw_model_outputs): # raw_model_outputs is of shape (HEAD_NUM, 87)
    # the raw output is a stack of outputs of different prediction heads,
    # and we need to rearrange it to obtain the final score,
    # therefore we need to initialize an empty matrix to gather theses values.
    model_outputs = torch.zeros(raw_model_outputs.shape[1])

    # the output of each head is a vector of length 11*2 + 21 + 11*4 = 87,
    # which corresponds to [11, 11, 21, 11, 11, 11, 11] for the 7 defects.
    SOFTMAX_INPUT_NUM = [11, 11, 21, 11, 11, 11, 11]

    # the score of exposure and white balance only uses the output of the
    # FIRST prediction head.
    split_point = sum(SOFTMAX_INPUT_NUM[:2])    # the bins of first two defects
    raw_model_outputs_first_head = raw_model_outputs[0]
    model_outputs[:split_point] = raw_model_outputs_first_head[:split_point]

    # other scores use the average of the outputs of all heads
    average_raw_model_outputs = torch.mean(raw_model_outputs, dim=0)
    model_outputs[split_point:] = average_raw_model_outputs[split_point:]

    # we split the output by the defects
    result = []
    sat_result = []
    n_pre = 0
    for n in range(OUTPUT_NUM):
        if n == SAT_IDX:    # if this is for saturation
            sat_result.append(model_outputs[n_pre:n_pre + SOFTMAX_INPUT_NUM[n]])
            n_pre += SOFTMAX_INPUT_NUM[n]
        else:
            result.append(model_outputs[n_pre:n_pre + SOFTMAX_INPUT_NUM[n]])
            n_pre += SOFTMAX_INPUT_NUM[n]

    # now result is nested list containing 6 lists of length 11,
    # and sat_result is a nested list containing a single list of length 11.
    # stack up the result into a torch tensor -> result.shape=(6, 11)
    result = torch.stack(result, axis=0)

    # also put the saturation result into a torch tensor -> result.shape=(1, 11)
    sat_result = torch.stack(sat_result, axis=0)

    # define constant values for computing the final score
    # the first one is an array:
    # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    peak = torch.range(0, 1, 0.1)

    # the second one is also an array, with negative values this time:
    # [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
    #   0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1.0]
    sat_peak = torch.range(-1, 1, 0.1)

    # compute the softmax of the raw scores (result.shape does not change)
    result = F.softmax(result)
    sat_result = F.softmax(sat_result)

    # compute the element-wise product (result.shape does not change)
    result = result * peak
    sat_result = sat_result * sat_peak

    # sum up on dimension-1, now result.shape=(6,) and sat_result.shape=(1,)
    result = result.sum(dim=1)
    sat_result = sat_result.sum(dim=1)

    # rearrange them to match with the final output order
    # the shape of the returned tensor is vector of length 7
    return torch.cat([result[:SAT_IDX], sat_result, result[SAT_IDX:]], dim=0)
