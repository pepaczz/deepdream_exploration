# Deepdream exploration

![Rainforest](https://raw.githubusercontent.com/pepaczz/deepdream_exploration/master/img_out/prales.jpg)

See [/codes/deepdream_explor_20180403a.ipynb](https://github.com/pepaczz/deepdream_exploration/blob/master/codes/deepdream_explor_20180403a.ipynb) for the main notebook. 

Due to a lot of visualizations the binary is rather large (18 MB) and reloading might be necessary to show it in GitHub. If problems prevail please [download the html](https://github.com/pepaczz/deepdream_exploration/blob/master/codes/deepdream_explor_20180403a.html) and open it on your computer.

______

This is an extension to the [Deepdream notebook from Tensorflow tutorials](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb).

Purpose of this notebook is to dive a bit deeper into the DeepDreaming technique as presented in the original tutorial. There are several main contributions on top of the original work:
 1. to present the readers how changing parameters affect resulting image
 2. to show that layers closer to the output capture more complicated patterns compared to layers deeper in the convolutional network
 3. to apply the effect on more pictures to see how identified patterns emerge on different parts of images depeding on their geometrical shape
 4. to put the functions into a separate file so that they can be used with more ease
