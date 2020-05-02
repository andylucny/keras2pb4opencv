# keras2pb4opencv
Convert Keras model to Tensorflow2 frozen graph to be loaded in OpenCV4

From OpenCV4.0 it is not so easy to use Keras models in .h5 format and 
from Tensorflow2 it is not so welknow how to save them into .pb nad .pbtxt.
This script solves all the troubles. Just some models has to be modified
little bit for OpenCV4, e.g. MobileNetV2, for which I have made this script,
requires to rename AddV2 to Add. These modifications can be performed also 
manually in .pbtxt without any change of .pb - therefore it is good to use 
both .pbtxt and .pb in OpenCV4, though using only .pb as a single parameter 
of cv2.dnn.readNet() would be enough, if this treatment is not needed.

The script is started like:

(keras) d:\emotion-models3> python keras2pb4opencv.py 7_mobileNet_v2_emotions.h5

and creates 7_mobileNet_v2_emotions.pbtxt and 7_mobileNet_v2_emotions.pb

It outputs also the key idea how use them in OpenCV4 where it is necessary
to pay much more attention to the shape of input which typically differs from
Keras. E.g. input (1,224,224,3) in Keras corresponds to (1,3,224,224) in OpenCV4.
Therefore the input image must be significantly rearranged.
