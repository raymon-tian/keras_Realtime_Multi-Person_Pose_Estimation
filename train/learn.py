from keras.layers import Input, Dense
from keras.models import Model
from keras.applications.vgg19 import VGG19

vgg19_model = VGG19(include_top=False,weights='imagenet')
for layer in vgg19_model.layers[1:11]:
    print (layer.name)
    w = layer.get_weights()
    print (w.shape)

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
print(model.summary())