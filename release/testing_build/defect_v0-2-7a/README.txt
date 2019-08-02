Testing specification for model defect_v0-2-7a

Input format:
  8-bit RGB of size 192x192 (resized via area interpolation, e.g. cv2.INTER_AREA, to 224x224, then center crop to 192x192),
  finally normalize pixel values to [0, 1].

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
