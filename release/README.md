## Components
- caffemodels: containing the converted checkpoints in caffe formats.
- prototxts: containing the generated prototxts.
- testing_build: containing package script and the final packages of all released versions.

## How to convert pytorch models to caffe models
* Install caffe and pycaffe.  
* Put `model_libs.py` under `[CAFFE_HOME]/python/caffe`.
* Use `python convert_model_to_caffe.py` to generate caffemodel and prototxt:
  * Modify the model implemented in PyCaffe to support new architectures.
  * Run the script until it waits for user prompt.
  * With reference to the examples under `./prototxt`, modify the generated `[ARCH_NAME].prototxt` and save it as `[ARCH_NAME]_caffe_test.prototxt`.
  * Resume the program and check if the testing error is acceptable (normally it should be zero). If so, you may proceed to the next step.
* Create the `rel.prototxt` used in final packaging by modifying`[ARCH_NAME]_caffe_test.prototxt`:
  * For every `Crop` layer:
    * Replace the content of its `crop_param ` with `crop_h: [HEIGHT] crop_w: [WIDTH]`, where the HEIGHT and WIDTH are defined in the dummy data layer that it refers to.
    * Change its bottom layer to the closest pooling layer.
  * Remove dummy data layers and all their references.
  * Replace all `ShuffleChannel` with `ChannelShuffle`, and all `shuffle_channel_param` with `channel_shuffle_param`.

## How to package
1. Set up the packaging environment.
2. Arrange the model files in order.
3. Use the package script (`testing_build/package.sh`).

For detailed instructions, please refer [here](http://note.youdao.com/noteshare?id=93964ab18a4907782fb5b3a39c7fbbfd).

## Problems
Please contact [Karen Wu](mailto:wuqiuhua@sensetime.com) or [Kaican Li](mailto:mjust.lkc@gmail.com).
