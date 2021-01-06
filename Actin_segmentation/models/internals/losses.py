from keras import backend as K
from keras.losses import binary_crossentropy, mean_absolute_error
import tensorflow as tf

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(y_true * y_pred, axis=-1)
    sum_ = K.sum(y_true + y_pred, axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice_coef(y_true, y_pred, smooth=1.):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    
    from wassname as well
    """
    intersection = K.sum(y_true * y_pred, axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

def bce_ssim_loss(y_true, y_pred):
    return DSSIM_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)

# code download from: https://github.com/bermanmaxim/LovaszSoftmax
def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


# --------------------------- BINARY LOSSES ---------------------------

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels

def lovasz_loss(y_true, y_pred):
    y_true, y_pred = K.cast(K.squeeze(y_true, -1), 'int32'), K.cast(K.squeeze(y_pred, -1), 'float32')
    #logits = K.log(y_pred / (1. - y_pred))
    logits = y_pred #Jiaxin
    loss = lovasz_hinge(logits, y_true, per_image = True, ignore = None)
    return loss

# Difference of Structural Similarity

def DSSIM_loss(y_true, y_pred, k1=0.01, k2=0.03, kernel_size=3, max_value=1.0):
    # There are additional parameters for this function
    # Note: some of the 'modes' for edge behavior do not yet have a
    # gradient definition in the Theano tree
    #   and cannot be used for learning
    
    c1 = (k1 * max_value) ** 2
    c2 = (k2 * max_value) ** 2

    kernel = [kernel_size, kernel_size]
    y_true = K.reshape(y_true, [-1] + list(K.int_shape(y_pred)[1:]))
    y_pred = K.reshape(y_pred, [-1] + list(K.int_shape(y_pred)[1:]))

    patches_pred = tf.extract_image_patches(y_pred, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")
    patches_true = tf.extract_image_patches(y_true, [1, 5, 5, 1], [1, 2, 2, 1], [1, 1, 1, 1], "SAME")

    # Reshape to get the var in the cells
    bs, w, h, c = K.int_shape(patches_pred)
    patches_pred = K.reshape(patches_pred, [-1, w, h, c])
    patches_true = K.reshape(patches_true, [-1, w, h, c])
    # Get mean
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    # Get variance
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    # Get std dev
    covar_true_pred = K.mean(patches_true * patches_pred, axis=-1) - u_true * u_pred

    ssim = (2 * u_true * u_pred + c1) * (2 * covar_true_pred + c2)
    denom = ((K.square(u_true)
              + K.square(u_pred)
              + c1) * (var_pred + var_true + c2))
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
    return K.mean((1.0 - ssim) / 2.0)

def dssim_mae_loss(y_true, y_pred):
    return DSSIM_loss(y_true, y_pred) + mean_absolute_error(y_true, y_pred)

#MSSim
#https://stackoverflow.com/questions/48744945/keras-ms-ssim-as-loss-function
def keras_SSIM_cs(y_true, y_pred):
    axis=None
    gaussian = make_kernel(1.5)
    x = tf.nn.conv2d(y_true, gaussian, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.conv2d(y_pred, gaussian, strides=[1, 1, 1, 1], padding='SAME')

    u_x=K.mean(x, axis=axis)
    u_y=K.mean(y, axis=axis)

    var_x=K.var(x, axis=axis)
    var_y=K.var(y, axis=axis)

    cov_xy=cov_keras(x, y, axis)

    K1=0.01
    K2=0.03
    L=1  # depth of image (255 in case the image has a differnt scale)

    C1=(K1*L)**2
    C2=(K2*L)**2
    C3=C2/2

    l = ((2*u_x*u_y)+C1) / (K.pow(u_x,2) + K.pow(u_x,2) + C1)
    c = ((2*K.sqrt(var_x)*K.sqrt(var_y))+C2) / (var_x + var_y + C2)
    s = (cov_xy+C3) / (K.sqrt(var_x)*K.sqrt(var_y) + C3)

    return [c,s,l]

def keras_MS_SSIM(y_true, y_pred):
    iterations = 5
    x=y_true
    y=y_pred
    weight = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    c=[]
    s=[]
    for i in range(iterations):
        cs=keras_SSIM_cs(x, y)
        c.append(cs[0])
        s.append(cs[1])
        l=cs[2]
        if(i!=4):
            x=tf.image.resize_images(x, (x.get_shape().as_list()[1]//(2**(i+1)), x.get_shape().as_list()[2]//(2**(i+1))))
            y=tf.image.resize_images(y, (y.get_shape().as_list()[1]//(2**(i+1)), y.get_shape().as_list()[2]//(2**(i+1))))
    c = tf.stack(c)
    s = tf.stack(s)
    cs = c*s

    #Normalize: suggestion from https://github.com/jorge-pessoa/pytorch-msssim/issues/2 last comment to avoid NaN values
    l=(l+1)/2
    cs=(cs+1)/2

    cs=cs**weight
    cs = tf.reduce_prod(cs)
    l=l**weight[-1]

    ms_ssim = l*cs
    ms_ssim = tf.where(tf.is_nan(ms_ssim), K.zeros_like(ms_ssim), ms_ssim)

    return K.mean(ms_ssim)

def mssim_mae_loss(y_true, y_pred):
    return keras_MS_SSIM(y_true, y_pred) + mean_absolute_error(y_true, y_pred)