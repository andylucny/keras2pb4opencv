# Load .h5 model and save it as .pb which can be loaded to OpenCV4
# based on https://github.com/leimao/Frozen_Graph_TensorFlow/blob/master/TensorFlow_v2/train.py
# and dcurt comment on https://github.com/opencv/opencv/issues/16399
# made by Andy Lucny www.agentspace.org/andy
import sys
import os
keras_model_path = sys.argv[1] if len(sys.argv) > 1 else 'keras_model.h5'
tf_model_name = os.path.splitext(sys.argv[1])[0] if len(sys.argv) > 1 else 'frozen_graph'
print('loading keras model')

# load Keras model
import tensorflow as tf
model = tf.keras.models.load_model(keras_model_path)
print('keras model loaded')

# convert Keras model to ConcreteFunction
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

# freeze the ConcreteFunction
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(full_model)

# get frozen graph
graph_def = frozen_func.graph.as_graph_def()
#layers = [op.name for op in frozen_func.graph.get_operations()]
#print("Frozen model layers: ")
#for layer in layers:
#    print(layer)
#print('Frozen model inputs:',frozen_func.inputs)
#print('Frozen model outputs:',frozen_func.outputs)
print('model converted to tf2 graph and frozen')

# modify the graph to be compatible with OpenCV4
# add modificiations for your network
# (you can try them also manually in .pbtxt without change of .pb)
# as an example these are modifications for MobileNetV2:
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'AddV2':
        graph_def.node[i].op = 'Add'

print('the frozen graph adjusted to OpenCV4')
print('(if it fails in OpenCV, play with .pbtxt)')

# save the frozen graph in the binary form including weights
tf.io.write_graph(graph_def,'',tf_model_name+'.pb',as_text=False)
print('weights saved as',tf_model_name+'.pb')

# remove weights
for i in reversed(range(len(graph_def.node))):
    if graph_def.node[i].op == 'Const':
        del graph_def.node[i]

# save the frozen graph without weights in the text form
tf.io.write_graph(graph_def,'',tf_model_name+'.pbtxt',as_text=True)
print('archtecture saved as',tf_model_name+'.pbtxt')

print()
print('now you can use it in OpenCV4, python or C++:')
print('import cv2')
print('net = cv2.dnn.readNet("frozen_graph.pbtxt","frozen_graph.pb")')

shape = [ dim for dim in model.input.shape]
strshape = '('
for i in range(len(shape)):
    strshape += ',' if i>0 else ''
    strshape += str(shape[0])
strshape += ')'

print('net.setInput(inputs) # inputs shape is '+strshape)
print('outputs = net.forward()')
