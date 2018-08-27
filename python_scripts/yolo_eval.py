def yolo_eval(yolo_outputs, image_shape, max_boxes=10, score_threshold=.3, iou_threshold=.5):
                
  """Evaluate YOLO model on given input batch and return filtered boxes."""
  box_xy, box_wh, box_confidence, box_class_probs = yolo_outputs
  boxes = yolo_boxes_to_corners(box_xy, box_wh)
  boxes, scores, classes = yolo_filter_boxes(
    boxes, box_confidence, box_class_probs, threshold=score_threshold)
  
  # Scale boxes back to original image shape.
  height = image_shape[0]
  width = image_shape[1]
  image_dims = K.stack([height, width, height, width])
  image_dims = K.reshape(image_dims, [1, 4])
  image_dims = K.cast(image_dims, dtype = tf.float64)
  boxes = boxes * image_dims
   
  # TODO: Something must be done about this ugly hack!
  max_boxes_tensor = K.variable(max_boxes, dtype='int32')
  K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
  boxes = K.cast(boxes, dtype = tf.float32)
  scores = K.cast(scores, dtype = tf.float32)
  nms_index = tf.image.non_max_suppression(
     boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
  boxes = K.gather(boxes, nms_index)
  scores = K.gather(scores, nms_index)
  classes = K.gather(classes, nms_index)
   
  return boxes, scores, classes
