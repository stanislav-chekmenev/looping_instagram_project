def yolo_head(feats, anchors, num_classes = 80):
  """Convert final layer features to bounding box parameters.
    Parameters
    ----------
    feats : tensor
        Final convolutional layer features.
    anchors : array-like
        Anchor box widths and heights.
    num_classes : int
        Number of target classes.
    Returns
    -------
    box_xy : tensor
        x, y box predictions adjusted by spatial location in conv layer.
    box_wh : tensor
        w, h box predictions adjusted by anchors and conv spatial resolution.
    box_conf : tensor
        Probability estimate for whether each box contains any object.
    box_class_pred : tensor
        Probability distribution estimate for each box over class labels.
    """
  num_anchors = len(anchors)
  # Reshape to batch, height, width, num_anchors, box_params.
  anchors_tensor = K.reshape(K.variable(anchors), [1, 1, 1, num_anchors, 2])
  
  #print(anchors_tensor.dtype)
  
  
  # Dynamic implementation of conv dims for fully convolutional model.
  conv_dims = K.shape(feats)[1:3]  # assuming channels last
  # In YOLO the height index is the inner most iteration.
  conv_height_index = K.arange(0, stop=conv_dims[0])
  conv_width_index = K.arange(0, stop=conv_dims[1])
  conv_height_index = K.tile(conv_height_index, [conv_dims[1]])

  # TODO: Repeat_elements and tf.split doesn't support dynamic splits.
  # conv_width_index = K.repeat_elements(conv_width_index, conv_dims[1], axis=0)
  conv_width_index = K.tile(
    K.expand_dims(conv_width_index, 0), [conv_dims[0], 1])
  conv_width_index = K.flatten(K.transpose(conv_width_index))
  conv_index = K.transpose(K.stack([conv_height_index, conv_width_index]))
  conv_index = K.reshape(conv_index, [1, conv_dims[0], conv_dims[1], 1, 2])
  conv_index = K.cast(conv_index, feats.dtype)

  feats = K.reshape(
    feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])
  conv_dims = K.cast(K.reshape(conv_dims, [1, 1, 1, 1, 2]), feats.dtype)


  box_xy = K.sigmoid(feats[..., :2])
  box_wh = K.exp(feats[..., 2:4])
  box_confidence = K.sigmoid(feats[..., 4:5])
  box_class_probs = K.softmax(feats[..., 5:])

  # Adjust preditions to each spatial grid point and anchor size.
  # Note: YOLO iterates over height index before width index.
  box_xy = (box_xy + conv_index) / conv_dims
  anchors_tensor = K.cast(anchors_tensor, dtype = tf.float64)
  #print("box_wh type is " + str(box_wh.dtype))
  box_wh = box_wh * anchors_tensor / conv_dims

  return box_xy, box_wh, box_confidence, box_class_probs
