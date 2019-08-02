Testing specification for model defect_v0-2-8b

Input format:
  8-bit RGB of size 224x224 (resized via bilinear interpolation to 640x640, then area interpolate, e.g. cv2.INTER_AREA, to 224x224),
  finally, normalize pixel values to [0, 1].

Output format:
  A vector of length 88, please following the guidelines in transfer.py to convert them to the final scores, which are:
    1. Bad Exposure
    2. Bad White Balance
    3. Bad Saturation
    4. Noise
    5. Haze
    6. Undesired Blur
    7. Bad Composition
  Please select the predictions according to demands.
