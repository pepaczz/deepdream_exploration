# Deepdream exploration

An extension to the [Deepdream notebook from Tensorflow tutorials](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb).

See [/codes/deepdream_explor_20180403a.ipynb](https://github.com/pepaczz/deepdream_exploration/blob/master/codes/deepdream_explor_20180403a.ipynb) for the main notebook.

Purpose of this notebook is to dive a bit deeper into the DeepDreaming technique as presented in the original tutorial. There are several main contributions on top of the original work:
 1. to present the readers how changing parameters affect resulting image
 2. to show that layers closer to the output capture more complicated patterns compared to layers deeper in the convolutional network
 3. to apply the effect on more pictures to see how identified patterns emerge on different parts of images depeding on their geometrical shape
 4. to put the functions into a separate file so that they can be used with more ease
