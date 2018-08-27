def yolo_filter_boxes(boxes, box_confidence, box_class_probs, threshold=.3):
  """Filter YOLO boxes based on object and class confidence."""
  box_scores = box_confidence * box_class_probs
  box_classes = K.argmax(box_scores, axis=-1)
  box_class_scores = K.max(box_scores, axis=-1)
  prediction_mask = box_class_scores >= threshold
  
  # TODO: Expose tf.boolean_mask to Keras backend?
  boxes = tf.boolean_mask(boxes, prediction_mask)
  scores = tf.boolean_mask(box_class_scores, prediction_mask)
  classes = tf.boolean_mask(box_classes, prediction_mask)
  return boxes, scores, classes