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

# 重みは [入力レイヤ(=特徴)の数, 出力レイヤ(=分類)の数] -> [2, 3]
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))

# 偏りは出力レイヤの数
b = tf.Variable(tf.zeros([3]))

# 神経網（グラフ）に重みと偏りを適用
L = tf.add(tf.matmul(X, W), b)
# ReLU関数を適用
L = tf.nn.relu(L)

# softmaxで結果値の合計を１にする。
# ex) [8.04, 2.76, -6.52] -> [0.53 0.24 0.23]
model = tf.nn.softmax(L)

# 予測値と実測値の乖離を計算する部分 cross entrophy を使った
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
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
