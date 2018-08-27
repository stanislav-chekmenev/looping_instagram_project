def get_classes_anchors(classes_path, anchors_path):
  with open(classes_path) as f:
    class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
  
  with open(anchors_path) as f:
    anchors = f.readline()
  anchors = [float(x) for x in anchors.split(',')]
  anchors = np.array(anchors).reshape(-1, 2)
  
  return class_names, anchors
