import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
'''
ratio = [0.5, 1, 2]
anchor_scales = [8, 16, 32]

anchor_base = np.zeros((len(ratio) * len(anchor_scales), 4), dtype=np.float32)

print(anchor_base)
subsample = 16 # 16 times subsampled from 800 px to 50
fmap_size = 50 # 800/16 feature map dimension
anchors = np.zeros((fmap_size * fmap_size * 9, 4))

center_x = subsample / 2
center_y = subsample / 2

for i in range(len(ratio)):
  for j in range(len(anchor_scales)):
    h = subsample * anchor_scales[j] * np.sqrt(ratio[i])
    w = subsample * anchor_scales[j] * np.sqrt(1 / ratio[i])

    index = i * len(anchor_scales) + j

    anchor_base[index, 0] = center_y - h / 2.
    anchor_base[index, 1] = center_x - w / 2.
    anchor_base[index, 2] = center_y + h / 2.
    anchor_base[index, 3] = center_x + w / 2.

print(anchor_base)

ctr_x = np.arange(16, (fmap_size+1) * 16, 16)
ctr_y = np.arange(16, (fmap_size+1) * 16, 16)

index = 0
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[x] - 8
        ctr[index, 0] = ctr_y[y] - 8
        index +=1
'''

'''
def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )
def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """


    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)




class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width, fmap_sizes = [4]):
        """Generates anchor boxes for all the feature maps of the feature pyramid.
        for example box.get_anchors(256,128) returns 6138 anchors

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.
          fmap_sizes: 2 ** value in fmap_sizes is the subsampling factor from img size to feature map size
        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in fmap_sizes
        ]
        return tf.concat(anchors, axis=0)
a_gen= AnchorBox()
anchors = a_gen.get_anchors(192, 192)
compute_iou(anchors[:10], anchors[4:5])
'''


"""
## Implementing utility functions

Bounding boxes can be represented in multiple ways, the most common formats are:

- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions
`[x, y, width, height]`

Since we require both formats, we will be implementing functions for converting
between the formats.
"""


def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.

    Arguments:
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.

    Returns:
      swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


"""
## Computing pairwise Intersection Over Union (IOU)

As we will see later in the example, we would be assigning ground truth boxes
to anchor boxes based on the extent of overlapping. This will require us to
calculate the Intersection Over Union (IOU) between all the anchor
boxes and ground truth boxes pairs.
"""


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


class AnchorBox:
    """Generates anchor boxes.

    This class has operations to generate anchor boxes for feature maps at
    strides `[8, 16, 32, 64, 128]`. Where each anchor each box is of the
    format `[x, y, width, height]`.

    Attributes:
      aspect_ratios: A list of float values representing the aspect ratios of
        the anchor boxes at each location on the feature map
      scales: A list of float values representing the scale of the anchor boxes
        at each location on the feature map.
      num_anchors: The number of anchor boxes at each location on feature map
      areas: A list of float values representing the areas of the anchor
        boxes for each feature map in the feature pyramid.
      strides: A list of float value representing the strides for each feature
        map in the feature pyramid.
    """

    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2 ** x for x in [0, 1 / 3, 2 / 3]]

        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(3, 8)]
        self._areas = [x ** 2 for x in [32.0, 64.0, 128.0, 256.0, 512.0]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        """Computes anchor box dimensions for all ratios and scales at all levels
        of the feature pyramid.
        """
        anchor_dims_all = []
        for area in self._areas:
            anchor_dims = []
            for ratio in self.aspect_ratios:
                anchor_height = tf.math.sqrt(area / ratio)
                anchor_width = area / anchor_height
                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis=-1), [1, 1, 2]
                )
                for scale in self.scales:
                    anchor_dims.append(scale * dims)
            anchor_dims_all.append(tf.stack(anchor_dims, axis=-2))
        return anchor_dims_all

    def _get_anchors(self, feature_height, feature_width, level):
        """Generates anchor boxes for a given feature map size and level

        Arguments:
          feature_height: An integer representing the height of the feature map.
          feature_width: An integer representing the width of the feature map.
          level: An integer representing the level of the feature map in the
            feature pyramid.

        Returns:
          anchor boxes with the shape
          `(feature_height * feature_width * num_anchors, 4)`
        """
        rx = tf.range(feature_width, dtype=tf.float32) + 0.5
        ry = tf.range(feature_height, dtype=tf.float32) + 0.5
        centers = tf.stack(tf.meshgrid(rx, ry), axis=-1) * self._strides[level - 3]
        centers = tf.expand_dims(centers, axis=-2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
        dims = tf.tile(
            self._anchor_dims[level - 3], [feature_height, feature_width, 1, 1]
        )
        anchors = tf.concat([centers, dims], axis=-1)
        return tf.reshape(
            anchors, [feature_height * feature_width * self._num_anchors, 4]
        )

    def get_anchors(self, image_height, image_width):
        """Generates anchor boxes for all the feature maps of the feature pyramid.
        for example box.get_anchors(256,128) returns 6138 anchors

        Arguments:
          image_height: Height of the input image.
          image_width: Width of the input image.

        Returns:
          anchor boxes for all the feature maps, stacked as a single tensor
            with shape `(total_anchors, 4)`
        """
        anchors = [
            self._get_anchors(
                tf.math.ceil(image_height / 2 ** i),
                tf.math.ceil(image_width / 2 ** i),
                i,
            )
            for i in [4]
        ]
        return tf.concat(anchors, axis=0)

def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes,
        having normalized coordinates.

    Returns:
      Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes


def resize_and_pad_image(
    image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
):
    """Resizes and pads image while preserving aspect ratio.

    1. Resizes images so that the shorter side is equal to `min_side`
    2. If the longer side is greater than `max_side`, then resize the image
      with longer side equal to `max_side`
    3. Pad with zeros on right and bottom to make the image shape divisible by
    `stride`

    Arguments:
      image: A 3-D tensor of shape `(height, width, channels)` representing an
        image.
      min_side: The shorter side of the image is resized to this value, if
        `jitter` is set to None.
      max_side: If the longer side of the image exceeds this value after
        resizing, the image is resized such that the longer side now equals to
        this value.
      jitter: A list of floats containing minimum and maximum size for scale
        jittering. If available, the shorter side of the image will be
        resized to a random value in this range.
      stride: The stride of the smallest feature map in the feature pyramid.
        Can be calculated using `image_size / feature_map_size`.

    Returns:
      image: Resized and padded image.
      image_shape: Shape of the image before padding.
      ratio: The scaling factor used to resize the image
    """
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    if jitter is not None:
        min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
    ratio = min_side / tf.reduce_min(image_shape)
    if ratio * tf.reduce_max(image_shape) > max_side:
        ratio = max_side / tf.reduce_max(image_shape)
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
    )
    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )
    return image, image_shape, ratio



def preprocess_data_ab(sample):
    """Applies preprocessing step to a single sample

    Arguments:
      sample: A dict representing a single training sample.

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    image = sample["image"]
    # bbox = swap_xy(sample["objects"]["bbox"])
    bbox = tf.cast(sample["objects"]["bbox"], dtype=tf.float32)

    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    # image, bbox = random_flip_horizontal(image, bbox)
    # image, image_shape, _ = resize_and_pad_image(image)
    image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
    bbox = tf.stack(
        [
            bbox[:, 0] *  1, #image_shape[1],
            bbox[:, 1] * 1, #image_shape[0],
            bbox[:, 2] * 1, #image_shape[1],
            bbox[:, 3] * 1, #image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id


class LabelEncoder:
    """Transforms the raw labels into targets for training.

    This class has operations to generate targets for a batch of samples which
    is made up of the input images, bounding boxes for the objects present and
    their class ids.

    Attributes:
      anchor_box: Anchor box generator to encode the bounding boxes.
      box_variance: The scaling factors used to scale the bounding box targets.
    """

    def __init__(self):
        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(
            [0.1, 0.1, 0.2, 0.2], dtype=tf.float32
        )

    def _match_anchor_boxes(
        self, anchor_boxes, gt_boxes, match_iou=0.5, ignore_iou=0.4
    ):
        """Matches ground truth boxes to anchor boxes based on IOU.

        1. Calculates the pairwise IOU for the M `anchor_boxes` and N `gt_boxes`
          to get a `(M, N)` shaped matrix.
        2. The ground truth box with the maximum IOU in each row is assigned to
          the anchor box provided the IOU is greater than `match_iou`.
        3. If the maximum IOU in a row is less than `ignore_iou`, the anchor
          box is assigned with the background class.
        4. The remaining anchor boxes that do not have any class assigned are
          ignored during training.

        Arguments:
          anchor_boxes: A float tensor with the shape `(total_anchors, 4)`
            representing all the anchor boxes for a given input image shape,
            where each anchor box is of the format `[x, y, width, height]`.
          gt_boxes: A float tensor with shape `(num_objects, 4)` representing
            the ground truth boxes, where each box is of the format
            `[x, y, width, height]`.
          match_iou: A float value representing the minimum IOU threshold for
            determining if a ground truth box can be assigned to an anchor box.
          ignore_iou: A float value representing the IOU threshold under which
            an anchor box is assigned to the background class.

        Returns:
          matched_gt_idx: Index of the matched object
          positive_mask: A mask for anchor boxes that have been assigned ground
            truth boxes.
          ignore_mask: A mask for anchor boxes that need to by ignored during
            training
        """
        iou_matrix = compute_iou(anchor_boxes, gt_boxes) # 1296 x 1 (mostly only 1 gt box and 12x12x9 anchorboxes)
        max_iou = tf.reduce_max(iou_matrix, axis=1)
        matched_gt_idx = tf.argmax(iou_matrix, axis=1)
        positive_mask = tf.greater_equal(max_iou, match_iou)
        negative_mask = tf.less(max_iou, ignore_iou)
        ignore_mask = tf.logical_not(tf.logical_or(positive_mask, negative_mask))
        return (
            matched_gt_idx,
            tf.cast(positive_mask, dtype=tf.float32),
            tf.cast(ignore_mask, dtype=tf.float32), iou_matrix
        )

    def _compute_box_target(self, anchor_boxes, matched_gt_boxes):
        """Transforms the ground truth boxes into targets for training"""
        box_target = tf.concat(
            [
                (matched_gt_boxes[:, :2] - anchor_boxes[:, :2]) / anchor_boxes[:, 2:],
                tf.math.log(matched_gt_boxes[:, 2:] / anchor_boxes[:, 2:]),
            ],
            axis=-1,
        )
        box_target = box_target / self._box_variance
        return box_target

    def _encode_sample(self, image_shape, gt_boxes, cls_ids):
        """Creates box and classification targets for a single sample"""
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2]) # cx, cy, w, h of the original img
        cls_ids = tf.cast(cls_ids, dtype=tf.float32)
        """
        positive_mask is 1 where iou bigger than some threshold. this serves as a label
        should be reshaped from list shape to tensor 12x12x9 and then to one-hot 
        encoding as per papers -> 12x12x18 (featuremap x featuremap x 2 * anchorboxes)
        
        """
        matched_gt_idx, positive_mask, ignore_mask, iou_mat = self._match_anchor_boxes(
            anchor_boxes, gt_boxes
        )
        pos_mask = np.reshape(positive_mask, newshape=[12,12,9])
        #matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
        #box_target = self._compute_box_target(anchor_boxes, matched_gt_boxes)
        #matched_gt_cls_ids = tf.gather(cls_ids, matched_gt_idx)
        #cls_target = tf.where(
        #    tf.not_equal(positive_mask, 1.0), -1.0, matched_gt_cls_ids
        #)
        #cls_target = tf.where(tf.equal(ignore_mask, 1.0), -2.0, cls_target)
        #cls_target = tf.expand_dims(cls_target, axis=-1)
        # dont add cls_target as not needed for our purpose
        # label = tf.concat([box_target, cls_target], axis=-1)
        #label = tf.concat([box_target], axis=-1)
        return positive_mask
        #return label

    def encode_batch(self, batch_images, gt_boxes, cls_ids):
        """Creates box and classification targets for a batch"""
        images_shape = tf.shape(batch_images)
        batch_size = images_shape[0]

        labels = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=True)
        for i in range(batch_size):
            label = self._encode_sample(images_shape, gt_boxes[i], cls_ids[i])
            labels = labels.write(i, label)
        # batch_images = tf.keras.applications.resnet.preprocess_input(batch_images)
        return batch_images, labels.stack()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # image_batch, bbox_batch, class_id_batch = []
    # # for sample in dataset:
    # image, bbox, class_id = preprocess_data(sample)
    # image_batch.append(image), bbox_batch.append(bbox), class_id_batch.append(class_id)
    # label_encoder = LabelEncoder()
    # batch_images, labels_stacked = label_encoder.encode_batch(tf.stack(image_batch), tf.stack(bbox_batch), tf.stack(class_id_batch))
    image_batch = []
    bbox_batch = []
    class_id_batch = []
    list_filenames = [f for f in os.listdir(r"C:\workspace\Chargrid\data\np_chargrid_unscaled_top_1h") if
                      os.path.isfile(os.path.join(r"C:\workspace\Chargrid\data\np_chargrid_unscaled_top_1h", f))]
    count = 0
    for example in list_filenames:
        print(example)
        image_batch = []
        bbox_batch = []
        class_id_batch = []
        img_dict = {}
        img_dict['objects'] = {}
        im = np.load(f"C:\workspace\Chargrid\data\\np_chargrid_unscaled_top_1h\\{example}")
        label = np.load(f"C:\workspace\Chargrid\data\\np_label_unscaled_top\\{example}")
        img_dict['image'] = im
        # twos[0] = y values, twos[1] = x values, empty if not found
        twos = np.where(label == 2)
        if len(twos[0]) != 0:
            img_dict['objects']['bbox'] = np.expand_dims(np.array([twos[1].min(), twos[0].min(), twos[1].max(), twos[0].max()], dtype=np.float), axis=0)
            img_dict['objects']['label'] = np.array([1])
            image, bbox, class_id = preprocess_data_ab(img_dict)
            image_batch.append(image)
            bbox_batch.append(bbox)
            class_id_batch.append(class_id)

            label_encoder = LabelEncoder()

            batch_images, batch_labels_stacked = label_encoder.encode_batch(tf.stack(image_batch), tf.stack(bbox_batch), tf.stack(class_id_batch))
            if np.sum(batch_labels_stacked[0]) > 0:
                np_labels = np.array(batch_labels_stacked)
                one_hot = False
                if one_hot:
                    np_labels = np.reshape(np_labels, [np_labels.shape[0], 1296, 1])
                    np_labels = np.where(np_labels == [0], [0, 1], [1, 0])
                    np_labels = np.reshape(np_labels, [np_labels.shape[0],1296 * 2, ])
                    np_labels= np.reshape(np_labels, [np_labels.shape[0], 12, 12, 18])
                    #label_encoder._anchor_box.get_anchors(192,192)
                    np.save(os.path.join(f"C:\workspace\Chargrid\data\\bbox_labels_cls_min_1_iou", example), np_labels[0])
                else:
                    np_labels = np.reshape(np_labels, [np_labels.shape[0], 1296, 1])
                    np_labels = np.reshape(np_labels, [np_labels.shape[0], 12, 12, 9])
                    np.save(os.path.join(f"C:\workspace\Chargrid\data\\bbox_labels_cls_min_1_iou_not_1_hot", example), np_labels[0])

    fig, (ax1, ax2, ) = plt.subplots(1, 2)
    ax1.imshow(im)
    # twos[1].min(), twos[0].min(), twos[1].max(), twos[0].max() (56, 73, 188, 115) xyxy ul, br
    ax1.add_patch(patches.Rectangle((56, 73), 188 - 56, 115 - 73, linewidth=1, edgecolor='r', facecolor='none'))
    ax2.imshow(label)
    plt.show()
    print()
    im = np.load(f"C:\workspace\Chargrid\data\\np_chargrid_unscaled_top_1h\\X00016469612.npy")
    im.shape
    plt.imshow(im.argmax(axis=2))
    im[100:110, 140:145]