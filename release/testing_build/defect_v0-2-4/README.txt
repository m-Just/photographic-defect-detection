Testing specification for model defect_v0-2-4

Input format:
  8-bit RGB of size 224x224 (resized via area interpolation, e.g. cv2.INTER_AREA),
  then normalize pixel values to [0, 1].

Output format:
  Seven floats representing the predicted scores for:
    1. Bad Exposure
    2. Bad White Balance
    3. Bad Saturation
    4. Noise
    5. Haze
    6. Undesired Blur
    7. Bad Composition
  Please select the predictions according to demands.
