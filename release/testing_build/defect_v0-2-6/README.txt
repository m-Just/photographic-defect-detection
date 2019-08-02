Testing specification for model defect_v0-2-6

Input format:
  8-bit RGB of size 224x224 (resized via area interpolation, e.g. cv2.INTER_AREA),
  then normalize pixel values to [0, 1].

Output format:
  Seven floats representing the raw scores (not the final scores) for:
    1. Bad Exposure
    2. Bad White Balance
    3. Noise
    4. Haze
    5. Undesired Blur
    6. Bad Composition
    7. Bad Saturation
  To convert them to the final scores and rearrange them in the correct order, please refer to the python code in transfer.py,
  then select the predictions according to demands.
