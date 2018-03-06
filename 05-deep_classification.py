# 毛と羽があるかどうかで哺乳類か鳥類かに分類する。
import tensorflow as tf
import numpy as np

# 特徴 [毛, 羽]
x_data = np.array(
                  [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# 分類 [その他, 哺乳類, 鳥類]
# one-hot encoded classification
y_data = np.array([
                   [1, 0, 0],  # その他
                   [0, 1, 0],  # 哺乳類
                   [0, 0, 1],  # 鳥類
                   [1, 0, 0],
                   [1, 0, 0],
                   [0, 0, 1]
                   ])

#########
# 神経網モデル
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#W, bの代わりに下記のコードを追加します。
W1 = tf.Variable(tf.random_uniform([2,10], -1., 1.)) #入力レイヤーの数, 隠しレイヤーの数
W2 = tf.Variable(tf.random_uniform([10,3], -1., 1.)) #隠しレイヤーの数、出力レイヤーの数

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

#Lの代わりに
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# softmaxで結果値の合計を１にする。
# ex) [8.04, 2.76, -6.52] -> [0.53 0.24 0.23]
model = tf.add(tf.matmul(L1,W2), b2)

# cost関数を変更する。
cost = tf.reduce_mean(
                      tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

#########
# 学習パート
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

#########
# テスト
# 0: その他 1: 哺乳類, 2: 鳥類
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('予測:', sess.run(prediction, feed_dict={X: x_data}))
print('実測:', sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('精度: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
