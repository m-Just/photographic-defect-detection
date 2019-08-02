## Components
- caffemodels: containing the converted checkpoints in caffe formats.
- prototxts: containing the generated prototxts.
- testing_build: containing package script and the final packages of all released versions.

## How to convert pytorch models to caffe models
1. Install caffe and pycaffe.  
2. Put `model_libs.py` under `[CAFFE_HOME]/python/caffe`.
3. Modify `python convert_model_to_caffe.py` to support any new architecture.
4. Run `python convert_model_to_caffe.py` and follow the instructions.
5. The generated caffemodel will be in `./caffemodels` and the prototxt will be in `./prototxts`.

## How to package
1. Set up the packaging environment.
2. Arrange the model files in order.
3. Use the package script (`testing_build/package.sh`).

For detailed instructions, please refer [here](http://note.youdao.com/noteshare?id=93964ab18a4907782fb5b3a39c7fbbfd).

## Problems
Please contact [Karen Wu](mailto:wuqiuhua@sensetime.com) or [Kaican Li](mailto:mjust.lkc@gmail.com).
