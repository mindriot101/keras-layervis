#!/usr/bin/env python


import matplotlib.pyplot as plt
from keras.models import Sequential
import numpy as np


class LayerVis(object):
    def __init__(self, model, *, verbose=False, n_columns=8, panel_size=3):
        self.model = model
        self.verbose = verbose
        self.n_columns = n_columns
        self.panel_size = panel_size

    def all_layers(self, image, *, othermodel=None):
        '''Visualise every layer of the model'''
        n_layers = len(self.model.layers)
        return self.all_up_to(image=image, last_layer=n_layers, othermodel=othermodel)
    
    def cumulative_layers(self, image, *, up_to=None, layer_filter=None, othermodel=None):
        '''Visualise a complete model by viewing the effect of each layer as they are added'''
        if layer_filter is not None:
            layer_filter = set(layer_filter)

        newmodel = Sequential()
        for i, layer in enumerate(self.model.layers):
            if up_to is not None and i >= (up_to - 1):
                break

            newmodel.add(layer)
            if layer_filter is None or layer.__class__ in layer_filter:
                self.all_layers(image, othermodel=newmodel)

    def all_up_to(self, image, last_layer, *, title=None, othermodel=None):
        '''Visualise a Keras model, combining the outputs of all the layers up to
        layer `last_layer`'''
        
        if othermodel is None:
            model = self.model
        else:
            model = othermodel
               
        if last_layer > (len(model.layers) + 1):
            raise IndexError('Cannot find layer index {} in model ({} layers)'.format(
                last_layer, len(model.layers)))

        submodel = Sequential()
        for layer in model.layers[:last_layer]:
            submodel.add(layer)

        if self.verbose:
            submodel.summary()

        prediction = np.squeeze(submodel.predict(np.expand_dims(image, axis=0)))

        if len(prediction.shape) not in {3, 4}:
            raise ValueError('Prediction shape ({}) does not suit images. Did you pick a dense layer?'
                             .format(prediction.shape))

        grayscale = len(prediction.shape) == 3

        if grayscale:
            n_plots = prediction.shape[2]
        else:
            n_plots = prediction.shape[3]

        ny = n_plots // self.n_columns

        assert n_plots % self.n_columns == 0
        fig, axes = plt.subplots(ny, self.n_columns, figsize=(self.n_columns * self.panel_size, ny * self.panel_size))
        axes = axes.ravel()
        for i in range(n_plots):
            ax = axes[i]
            if grayscale:
                _show_image(prediction[:, :, i], ax)
            else:
                _show_image(prediction[:, :, :, i], ax)
        for ax in axes:
            ax.axis('off')

        if title is not None:
            fig.suptitle(title)
        fig.subplots_adjust(wspace=0.02, hspace=0.02)

        return fig

def _show_image(data, ax=None):
    '''Helper function to display an image on a matplotlib axis'''
    ax = ax if ax is not None else plt.gca()
    ax.imshow(np.squeeze(data), cmap='gray')
    

            
            
# if __name__ == '__main__':
#     from keras.models import load_model, Sequential
#     from keras.layers import Conv2D

#     model = Sequential([
#         Conv2D(32, (3, 3), input_shape=(224, 224, 1)),
#         Conv2D(32, (11, 11), input_shape=(224, 224, 1)),
#     ])

#     vis = LayerVis(model, verbose=True)

#     image_data = np.random.uniform(0., 1., (224, 224, 3))
#     vis.
