
# Object Detection Module

## Requirements
* ffmpeg 

## Module Test

### Operation Test

#### Image path processing
  
#### Video processing

#### Server mode
  
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

## Module Test Result
| category | Description | Result |
| ---- | ----------- | ------ |
| image processing operation | image path 에 대해 object detection 을 수행 | Y |
| aaa |          | d |

