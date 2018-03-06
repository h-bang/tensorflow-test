import tensorflow as tf

# mnistサンプルはtensorflowにすでに入っているのでimportするだけで簡単に使える。
from tensorflow.examples.tutorials.mnist import input_data

# これを実行するとデーターセットを自動でダウンロードして読み取る。one_hot オプションを指定
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 神経網モデル構成
######

# 入力レイヤーは28x28pixelなので784にする。
X = tf.placeholder(tf.float32, [None, 784])
# 出力レイヤーは0~9までの数字なので10にする・
Y = tf.placeholder(tf.float32, [None, 10])

# 隠しレイヤーのサイズは256にしてみる。
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))

L1 = tf.nn.relu(tf.matmul(X, W1))

# 2番目の隠しレイヤを追加し、サイズは256にする。
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))

L2 = tf.nn.relu(tf.matmul(L1, W2))

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))

model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#########
#　学習させる
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# データーサイズが大きいので少しずつ学習させる。（100個ずつ）
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0
    
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val
    
    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('学習完了!')

######
# 結果確認
######

# modelで予測した値と実際のレーブルであるYを比較
# tf.argmax 関数で一番確率の高い数字を予測と評価する。
# 例) [0.1 0 0 0.7 0 0.2 0 0 0 0] -> 3
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('精度:', sess.run(accuracy,
                      feed_dict={X: mnist.test.images,
                      Y: mnist.test.labels}))
