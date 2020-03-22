
# Object Detection Module

## description
* Object detection 을 수행하는 모듈이다.
* 제공하는 algorithm
  * YOLO34
    * CPU version 과 GPU version 을 모두 지원한다.
* 3 개의 동작 모드를 제공한다.
  * image path processing
  * video file processing
  * image processing server

## Next Steps
* pydarknet 의 image input format 은 RGB 가 이날 BGR 이다.   
  이에 대한 성능 평가를 진행해야 한다. (200322)

## Requirements
### Python Packages
  $ conda create --name visual python=3.7.3   
  $ conda activate visual   
  $ pip install opencv-python==4.1.1.26   
  $ pip install opencv-contrib-python==4.1.1.26   
  $ pip install pillow==7.0.0   
  $ pip install matplotlib==3.2.1   
  $ pip install psutil==5.6.7
  $ pip install moviepy==1.0.1    
  $ pip install yolo34py-gpu   
* YOLO34 python package.   
      Refer to https://pypi.org/project/yolo34py-gpu/   
* It is noted that yolo34 python package doesn't support MAC OS.   
      Refer to https://github.com/madhawav/YOLO3-4-Py/issues/41

### System Configuration
* YoloGpuPackage 를 사용할 경우,
  libdarknet.so 폴더가 LD_LIBRARY_PATH 에 정의되어 있어야 한다.

## Module Server Interface Protocol

[Click here](https://github.com/neohoon/ObjectDetection/wiki/Module-Server-Interface-Protocol)

## Module Test

### Operation Test

#### Image Path Processing    
* Image path 폴더에 있는 모든 이미지 파일을 처리한다.

#### Video File processing
* Video file 을 처리한다.

#### Image Processing Server


### Parameter Test

#### detect_height parameter
* Object detection 함수로 입력되는 이미지의 크기를 고정하는 변수이다.
* 매우 큰 이미지가 들어올 경우, 계산 시간을 줄이기 위해 적당한 이미지로 resize 한 이후 Object_Detection 을 수행한다.
* Object detection 결과는 변경된 이미지에 대한 결과이므로, 이를 다시 원 이미지 크기에 맞춰 재조정되는지 확인해야 한다.
* ini['OBJECT_DETECCT']['detect_height'] 값을 변화시켜 가면서 object detection 결과를 확인해야 한다.

#### roi parameter
* Object detection 을 수행하는 영역을 정의하는 변수이다.
* 이를 조정해 가면서 object detection 이 그 영역에서만 동작하는지 확인해야 한다.
* ini['OBJECT_DETECT']['roi'] 값을 변화시켜 가면서 object detection 결과를 확인해야 한다.

## Module Test Summary
| category | Task | Result | Remark |
| -------- | -----| ----------- | ------ |
| Operation test | Image path processing   | Y |  |
| Operation test | Video path processing   | Y |  |
| Operation test | Image processing server | Y |  |
