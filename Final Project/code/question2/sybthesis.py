import numpy as np
import time
import scipy
from PIL import Image
from os import listdir, mkdir, getcwd
from keras.applications.vgg16 import VGG16
from keras import backend as K

# model weights and constants
Content_weight = 0.021
style_weight = 7.6
total_variation_weight = 1.5
iterations = 9

limit = 420  # limit the size of images
height = limit
width = limit

# loss functions
def Content_loss(Content, whole):
    return K.sum(K.square(whole - Content))

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, whole):
    S = gram_matrix(style)
    C = gram_matrix(whole)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = K.square(x[:, :width-1, :height-1, :] - x[:, 1:, :height-1, :])
    b = K.square(x[:, :width-1, :height-1, :] - x[:, :width-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

def total_loss(model, whole_image):
    loss = K.variable(0.)

    layers = dict([(layer.name, layer.output) for layer in model.layers])
    layer_features = layers['block1_conv2'] # Content loss layers
    Content_image_features = layer_features[0, :, :, :]
    whole_features = layer_features[2, :, :, :]
    # Content loss
    loss += Content_weight * Content_loss(Content_image_features,
                                          whole_features)
    # style loss layers
    feature_layers = ['block1_conv2', 'block2_conv2',
                      'block3_conv3', 'block4_conv3',
                      'block5_conv3']

    for layer_name in feature_layers:
        layer_features = layers[layer_name]
        style_features = layer_features[1, :, :, :]
        whole_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, whole_features)
        loss += (style_weight / len(feature_layers)) * sl

    # total variation loss
    loss += total_variation_weight * total_variation_loss(whole_image)
    return loss # total loss


def eval_loss_and_grads(x):
    x = x.reshape((1, width, height, 3))
    outputs = [loss]
    outputs += grads
    f_outputs = K.function([whole_array], outputs)
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

def minimize_loss(whole):
    x = np.random.uniform(0, 255, (1, width, height, 3)) - 128
    evaluator = Evaluator()

    print("\n\nProcessing: " + whole)
    for i in range(iterations):

        # print diagnostic information
        print('Start of iteration', i)
        start_time = time.time()
        x, min_val, info = scipy.optimize.fmin_l_bfgs_b(evaluator.loss,
                                                        x.flatten(),
                                                        fprime=evaluator.grads,
                                                        maxfun=20)
        print('Current loss value:', min_val)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))
    return x

# Evaluator class
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

# util functions

def convert_to_image(img_array):
    img_array = img_array.reshape((width, height, 3))  # reshape
    img_array = img_array[:, :, ::-1]
    img_array[:, :, 0] += 103.939
    img_array[:, :, 1] += 116.779
    img_array[:, :, 2] += 123.68
    img_array = np.clip(img_array, 0, 255).astype('uint8')

    return Image.fromarray(img_array)  # return image

def prelim_img_process(image):
    image = image.resize((height, width), Image.ANTIALIAS)  # resize
    image = np.asarray(image, dtype='float32')  # cast to np array
    image = np.expand_dims(image, axis=0)  # add placeholder dimension

    image = image[:,:,:,:3] # remove alpha chanel
    image[:, :, :, 0] -= 103.939  # RGB values obtained from ImageNet
    image[:, :, :, 1] -= 116.779
    image[:, :, :, 2] -= 123.68
    image = image[:, :, :, ::-1]
    return image

def search_img_names():
    all_Content_names = listdir("ContentImages")
    all_style_names = listdir("styleImages")
    return all_Content_names, all_style_names

def search_Content_img(name):
    image = Image.open('./ContentImages/' + str(name)
                       )  # search the Content image
    # change the global variable to fit center image
    global height,width
    width = image.size[1]
    height = image.size[0]
    # compress
    if(height>limit or width>limit):
        if(height>width):
            ratio = height/limit
        else:
            ratio = width/limit
        height = int(height/ratio)
        width = int(width/ratio)

    image = prelim_img_process(image)
    return image

def search_style_img(name):
    image = Image.open('./styleImages/' + str(name)
                       )  # search the style image

    # rotate style image to minish the loss on style images
    if (image.size != (width,height)
        and (image.size[1] >= image.size[0]) != (height >= width)):
        image = image.rotate(90) # rotate style-image

    image = prelim_img_process(image)
    return image
#-----------------------------------------------------------------------------

if __name__ == "__main__":

    # search all images references
    Content_names, style_names = search_img_names()

    # process each Content image
    for Content_name in Content_names:
        # create intermediate directory associated output
        mkdir('./outputImages/' + Content_name[:-4])
        content_images = search_Content_img(Content_name)

        # create all style wholes for the current Content image
        for style_name in style_names:
            style_images = search_style_img(style_name)

            # create whole name and save destination
            whole = Content_name[:-4] + style_name[:-4]
            save_dir = getcwd() + '/outputImages/' + \
                Content_name[:-4] + '/' + whole

            # create placeholder image, used to store merger image
            whole_array = K.placeholder((1, width,height, 3))

            # concatenate the image arrays
            input_tensor = K.concatenate([content_images,
                                          style_images,
                                          whole_array], axis=0)

            # load model, iteratively merge and consolidate the two images
            # load the model
            model = VGG16(input_tensor=input_tensor,
                          weights='imagenet', include_top=False)
            # calculate whole loss
            loss = total_loss(model, whole_array)
            # calulate gradients of generated image
            grads = K.gradients(loss, whole_array)
            # run optimization using previously calculated loss values
            x = minimize_loss(whole)
            # convert and finalize np array
            final = convert_to_image(x)
            # save final rendition appropriately
            final.save(save_dir + '.jpeg', "jpeg")