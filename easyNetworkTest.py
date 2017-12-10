import tensorflow as tf
from skimage import io

saver = tf.train.import_meta_graph("save/easyModel.ckpt.meta")

with tf.Session() as sess:
    saver.restore(sess, "save/easyModel.ckpt")
    image = io.imread("test2/test.png",as_grey=True)
    io.imshow(image)
    image = image.reshape(1, 784)
    _input = tf.get_default_graph().get_tensor_by_name('input:0')
    confirm = tf.get_default_graph().get_tensor_by_name('confirm:0')
    result = sess.run(confirm, feed_dict={_input: image})
    print(result[0])
    print("识别结果为", result[0].argmax())
