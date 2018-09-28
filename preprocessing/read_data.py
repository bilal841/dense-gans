import tensorflow as tf

def tf_record_parser(record):
    keys_to_features = {
        'annotation': tf.FixedLenFeature([], tf.string),
        "height": tf.FixedLenFeature((), tf.int64),
        "width": tf.FixedLenFeature((), tf.int64)
    }

    features = tf.parse_single_example(record, keys_to_features)

    annotation = tf.decode_raw(features['annotation_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    # reshape input and annotation images
    annotation = tf.reshape(annotation, (height,width,1), name="annotation_reshape")
    annotation = tf.to_float(annotation)

    return annotation