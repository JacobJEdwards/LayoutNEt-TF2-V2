from typing import Iterator

import tensorflow as tf
import numpy as np
import config
from PIL import Image


class Dataset:
    def __init__(self) -> None:
        # prepare dataset
        _dataset = tf.data.TFRecordDataset(config.filenamequeue)
        _dataset = _dataset.map(self._decode_tfrecords)

        _dataset = _dataset.shuffle(buffer_size=1000,
                                    reshuffle_each_iteration=True)
        _dataset = _dataset.repeat()
        _dataset = _dataset.batch(batch_size=config.batch_size)
        _dataset = _dataset.as_numpy_iterator()

        self.dataset: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = _dataset

    @staticmethod
    def _decode_tfrecords(example_string: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        features = tf.io.parse_single_example(
            example_string,
            features={
                "label": tf.io.FixedLenFeature([], tf.int64),
                "textRatio": tf.io.FixedLenFeature([], tf.int64),
                "imgRatio": tf.io.FixedLenFeature([], tf.int64),
                'visualfea': tf.io.FixedLenFeature([], tf.string),
                'textualfea': tf.io.FixedLenFeature([], tf.string),
                "img_raw": tf.io.FixedLenFeature([], tf.string)
            })

        _image = tf.io.decode_raw(features['img_raw'], tf.uint8)
        _image = tf.reshape(_image, [60, 45, 3])
        _image = tf.cast(_image, tf.float32)

        _resized_image = tf.image.resize_with_crop_or_pad(_image, 64, 64)
        _resized_image = _resized_image / 127.5 - 1.

        _label = tf.cast(features['label'], tf.int32)

        _text_ratio = tf.cast(features['textRatio'], tf.int32)
        _img_ratio = tf.cast(features['imgRatio'], tf.int32)

        _visual_fea = tf.io.decode_raw(features['visualfea'], tf.float32)
        _visual_fea = tf.reshape(_visual_fea, [14, 14, 512])

        _textual_fea = tf.io.decode_raw(features['textualfea'], tf.float32)
        _textual_fea = tf.reshape(_textual_fea, [300])

        return _resized_image, _label, _text_ratio, _img_ratio, _visual_fea, _textual_fea

    def next(self):
        return next(self.dataset)


if __name__ == '__main__':
    dataset = Dataset()
    resized_image, label, textRatio, imgRatio, visualfea, textualfea = dataset.next()

    # we have 128 images
    h = w = 64
    canva = np.zeros((h * 16, w * 8, 3))
    for idx, image in enumerate(resized_image):
        i = idx % 8
        j = idx // 8
        canva[j * h:j * h + h, i * w:i * w + w] = (image + 1) / 2
    
    img = Image.fromarray(np.uint8(canva * 255))
    img.save('batch.png')
