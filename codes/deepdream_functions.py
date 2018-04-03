import tensorflow as tf
import numpy as np
from io import BytesIO
import PIL.Image
from IPython.display import clear_output, Image, display, HTML
from functools import partial
import matplotlib.pyplot as plt


#def img_noise(w=224, h=224):
def img_noise(w=400, h=400):
    return np.random.uniform(size=(h,w,3)) + 100.0

# Helper functions for TF Graph visualization

# removes large constants from the graph
def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add() 
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>"%size)
    return strip_def
  
def rename_nodes(graph_def, rename_func):
    res_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = res_def.node.add() 
        n.MergeFrom(n0)
        n.name = rename_func(n.name)
        for i, s in enumerate(n.input):
            n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
    return res_def
  
def show_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))
  
    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))

    
#####################################################

def showarray(a, fmt='jpeg', render_image=True):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    if render_image:
        display(Image(data=f.getvalue()))
    else:
        return a
    
def show_save_array(a, save_path=None, fmt='jpeg', render_image=True):
    a = np.uint8(np.clip(a, 0, 1)*255)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    
    if render_image:
        display(Image(data=f.getvalue()))
    if save_path is not None:
        PIL.Image.fromarray(a).save(save_path)
        
#         img_to_save=Image(data=f.getvalue())
#         img_to_save.save(save_path)
        
def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

def T(layer,graph):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("import/%s:0"%layer)

def render_naive(t_obj, t_input, sess, img0=None, iter_n=20, step=1.0):
    
    # generate noise image if no input image is supplied
    if img0 is None:
        img0 = img_noise()
    
    # Computes the mean of elements across dimensions of a tensor
    # i.e. for that single feature map it decides which columns and rows contain 
    #      the most 'active' cells
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
       
    # tf.gradients(ys, xs, ...)
    # Constructs symbolic derivatives of sum of ys w.r.t. x in xs.
    # It returns a list of Tensor of length len(xs) where each tensor is the sum(dy/dx) for y in ys
    # -> derivative of t_input (noise image) w.r.t. t_score, where t_input is the noise image?
    # 
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    img = img0.copy()
    # img = img0 # newly added
    
    for i in range(iter_n):
        
        # method run(fetches, feed_dict=None, options=None, run_metadata=None)
        #g, score = sess.run([t_grad, t_score], {t_input:img0})
        g, score = sess.run([t_grad, t_score], {t_input:img})
        
        # normalizing the gradient, so the same step size should work 
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
        print(score, end = ' ')
#     clear_output()
    showarray(visstd(img))
    
    
################################################################

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]

resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(t_input, img, t_grad, sess, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over 
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

#########################################################

def render_multiscale(t_obj, t_input, sess, img0=None, iter_n=10, step=1.0, octave_n=3, octave_scale=1.4):
    
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    
    # generate noise image if no input image is supplied
    if img0 is None:
        img0 = img_noise()
    
    img = img0.copy()
    # img = img0 # newly added
    
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(t_input, img, t_grad, sess)
            # normalizing the gradient, so the same step size should work 
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
            print('.', end = ' ')
        clear_output()
        showarray(visstd(img))
        
########################################################

def create_k5(): 
    k = np.float32([1,4,6,4,1])
    k = np.outer(k, k)
    k5x5 = k[:,:,None,None]/k.sum()*np.eye(3, dtype=np.float32)
    return(k5x5)
    
def lap_split(img, k5x5):
    '''Split the image into lo and hi frequency components'''
    with tf.name_scope('split'):
        lo = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
        lo2 = tf.nn.conv2d_transpose(lo, k5x5*4, tf.shape(img), [1,2,2,1])
        hi = img-lo2
    return lo, hi

def lap_split_n(img, n, k5x5):
    '''Build Laplacian pyramid with n splits'''
    levels = []
    for i in range(n):
        img, hi = lap_split(img, k5x5)
        levels.append(hi)
    levels.append(img)
    return levels[::-1]

def lap_merge(levels, k5x5):
    '''Merge Laplacian pyramid'''
    img = levels[0]
    for hi in levels[1:]:
        with tf.name_scope('merge'):
            img = tf.nn.conv2d_transpose(img, k5x5*4, tf.shape(hi), [1,2,2,1]) + hi
    return img

def normalize_std(img, eps=1e-10):
    '''Normalize image by making its standard deviation = 1.0'''
    with tf.name_scope('normalize'):
        std = tf.sqrt(tf.reduce_mean(tf.square(img)))
        return img/tf.maximum(std, eps)

def lap_normalize(img, k5x5, scale_n=4):
    '''Perform the Laplacian pyramid normalization.'''
    img = tf.expand_dims(img,0)
    tlevels = lap_split_n(img, scale_n, k5x5)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels, k5x5)
    return out[0,:,:,:]

################################################################

def render_lapnorm(t_obj, t_input, sess, img0=None, visfunc=visstd,
                   iter_n=10, step=1.0, octave_n=3, octave_scale=1.4, lap_n=4,
                  render_image=True, verbose=True, noise_img_size=244):
    
    k5x5 = create_k5()
    
    # generate noise image if no input image is supplied
    if img0 is None:
        img0 = img_noise(w=noise_img_size,h=noise_img_size)
    
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!
    # build the laplacian normalization graph
    lap_norm_func = tffunc(np.float32)(partial(lap_normalize, k5x5=k5x5, scale_n=lap_n))

    img = img0.copy()
    # img = img0 # newly added
    
    for octave in range(octave_n):
        if octave>0:
            hw = np.float32(img.shape[:2])*octave_scale
            img = resize(img, np.int32(hw))
        for i in range(iter_n):
            g = calc_grad_tiled(t_input, img, t_grad, sess)
            g = lap_norm_func(g)
            img += g*step
            if verbose:
                print('.', end = ' ')
        
        if render_image:
            clear_output()
            showarray(visfunc(img))
    
    if not render_image:
        return showarray(visfunc(img), render_image=render_image)

# my_img = render_lapnorm(T(layer)[:,:,:,channel], render_image=False)

################################################################

def render_deepdream(t_obj, t_input, sess, graph=None, img0=None, 
                     iter_n=10, step=1.5, octave_n=4, octave_scale=1.4, save_path=None, render_image=True):
    '''
    Takes a supplied image (or a noise) and applies filters from selected inception module.
    
    arguments:
    t_obj: either id of an inceptrion layer, e.g. 'mixed4c' OR tf object generated from this layer
    t_input, sess: tensorflow-related arguments
    grapg: should be supplied only in case when t_obj is string (not a tf object)
    img0: image the deepdream should be applied on. If None (default), then random noise is used
    iter_n, step, octave_n, octave_scale: refer to render_deepdream function
    save_path: Saves the image to path unless None (default)
    render_image: if False then returns the numpy image array. Default True
    '''
    
    # if t_obj is supplied as sting with inception layer ID, than it converts to the tf object 
    if isinstance(t_obj, str):
        t_obj = tf.square(T(t_obj, graph))
        
    # generate noise image if no input image is supplied
    if img0 is None:
        img0 = img_noise()
    
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

    # split the image into a number of octaves
    #img = img0
    img = img0.copy()
    
    octaves = []
    for i in range(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)
    
    # generate details octave by octave
    for octave in range(octave_n):
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in range(iter_n):
            g = calc_grad_tiled(t_input, img, t_grad, sess)
            img += g*(step / (np.abs(g).mean()+1e-7))
            print('.',end = ' ')
        clear_output()
#         showarray(img/255.0)
        if render_image:
            show_save_array(img/255.0, save_path=save_path, fmt='jpeg')
    
    # return the image if not for rendering
    if not render_image:
        
        # also save it if required
        if save_path is not None:
            show_save_array(img/255.0, save_path=save_path, fmt='jpeg', render_image=False)
        
        return showarray(img/255.0, render_image=False)
    
    
        # return img
    
################################################################    

def load_render_deepdream(t_obj, t_input, sess, graph, img0=None,
                          iter_n=10, step=1.5, octave_n=4, octave_scale=1.4, 
                          save_path=None, orig_image_only=False, render_image = True):
    '''
    Wrapper around the render_deepdream function.
    Loads the image and either shows it as it is (orig_image_only=True) or renders it as deepdream.
    
    arguments:
    t_obj: id of an inceptrion layer, e.g. 'mixed4c'
    t_input, sess, graph: tensorflow-related arguments
    img0: image the deepdream should be applied on. If None (default), then random noise is used
    iter_n, step, octave_n, octave_scale: refer to render_deepdream function
    save_path: Saves the image to path unless None (default)
    orig_image_only: if True, then render original image only whitout any modifications. Default False
    render_image: if False then returns the numpy image array. Default True
    '''
    
    # generate noise image if no input image is supplied
    if img0 is None:
        img0 = img_noise()
    else:
        img0 = PIL.Image.open(img0)
        img0 = np.float32(img0)
        
    img=img0.copy()
    
    if render_image:
        # either render image itself or as a deepdream
        if orig_image_only:
            showarray(img0/255.0)
            
        else:
            render_deepdream(tf.square(T(t_obj, graph)), t_input, sess, img0=img, 
                             iter_n=iter_n, step=step, octave_n=octave_n, octave_scale=octave_scale,
                             save_path=save_path)
    else:
        if orig_image_only:
            return showarray(img0/255.0, render_image=False)
        else:
            return render_deepdream(tf.square(T(t_obj, graph)), t_input, sess, img0=img, 
                                    iter_n=iter_n, step=step, octave_n=octave_n, octave_scale=octave_scale,
                                    save_path=save_path, render_image=False)
        
        
def plot_inception_module(graph, t_input, sess, inception_module_id='mixed4a', map_numbers=1, noise_img_size=224):
    '''
    Creates panel of six plot showing maps in different layers within an inception module.
    
    arguments:
    graph, t_input, sess: tensorflow-related arguments
    inception_module_id: id of inception module, default 'mixed4a'
    map_numbers: id of feature map to be taken for each layer
        if int, then uses this feature map number
        if list of ints, then specific map is specified for each layer
    '''

    # for debug purposes only
    #inception_module_id = 'mixed4a'
    #map_number = 15
    
    list_of_feature_maps = []
    inception_module_captions =['1x1_pre_relu', '3x3_bottleneck_pre_relu', '3x3_pre_relu', 
                                '5x5_bottleneck_pre_relu', '5x5_pre_relu', 'pool_reduce_pre_relu']

    # generate list of map_number for each layer
    if isinstance(map_numbers,int):
        map_numbers = [map_numbers] * len(inception_module_captions)
    if len(map_numbers) < len(inception_module_captions):
        map_numbers = [map_numbers[0]] * len(inception_module_captions)
    
    # print('Feature map numbers: ' + map_numbers)
    
    # iterate over ineption module layers
    for j, layer_caption in enumerate(inception_module_captions):

        # extend the list of feature maps
        list_of_feature_maps.append(T(inception_module_id+'_'+layer_caption,graph)[:,:,:,map_numbers[j]])

    # define figure
    fig = plt.figure(figsize=(15, 12))
    plt.suptitle(inception_module_id)
    
    # iterae over all feature maps
    for i, feature_map in enumerate(list_of_feature_maps):

        # save rendered feature map
        print(str(i+1)+'/'+str(len(list_of_feature_maps))+'   ', end="")
        rendered_feature_map = render_lapnorm(feature_map, t_input, sess, render_image=False, verbose=False,
                                             noise_img_size=noise_img_size)

        a=fig.add_subplot(2,3,i+1)
        plt.imshow(rendered_feature_map)
        plt.axis('off')
        plt.tight_layout()
        plt.title(str(inception_module_captions[i]) + ', feature map ' + str(map_numbers[j]))
        
        
def render_deepdream_w_original(t_obj, t_input, sess, graph,
                                img0=None, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4,
                                save_path=None):
    '''
    Wrapper around the load_render_deepdream function.
    Renders image with deepdream modification as well as its unmodified version for comparison
    
    arguments:
    t_obj: id of an inceptrion layer, e.g. 'mixed4c'
    t_input, sess, graph: tensorflow-related arguments
    img0: image the deepdream should be applied on. If None (default), then random noise is used
    iter_n, step, octave_n, octave_scale: refer to render_deepdream function
    save_path: Saves the image to path unless None (default)
    orig_image_only: if True, then render original image only whitout any modifications. Default False
    render_image: if False then returns the numpy image array. Default True
    '''
    
    plt.rcParams["figure.figsize"] = (15,6)

    im_orig =load_render_deepdream(t_obj=t_obj, t_input=t_input, sess=sess, graph=graph, 
                                   img0=img0, save_path=None, orig_image_only=True, render_image = False)
    im_deep = load_render_deepdream(t_obj=t_obj, t_input=t_input, sess=sess, graph=graph, 
                                    img0=img0, save_path=save_path, render_image = False)

    fig = plt.figure()
    ax2 = fig.add_axes([.3,  0,.7, 1])
    ax1 = fig.add_axes([ 0,.45,.4,.6])

    ax1.imshow(im_orig)
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    
    ax2.imshow(im_deep)
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax2.set_xticklabels([])
    ax2.set_xticks([])