# %%
import subprocess
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import glob
from src import facenet  # パス変更
import tensorflow as tf
import numpy as np
from PIL import Image


class FaceEmbedding(object):
    def __init__(self, model_path):
        # モデルを読み込んでグラフに展開
        facenet.load_model(model_path)
        # リサイズする縦・横の値
        self.input_image_size = 150
        self.sess = tf.Session()
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph(
        ).get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]

    def __del__(self):
        self.sess.close()

    def load_image(self, image_path, width, height, mode):
        image = Image.open(image_path)
        # 150x150にリサイズ
        image = image.resize([width, height], Image.BILINEAR)
        # RGBに変換
        return np.array(image.convert(mode))

    def face_embeddings(self, image_path):
        image = self.load_image(
            image_path, self.input_image_size, self.input_image_size, 'RGB')
        # prewhiten()
        # -> 平均値を差し引き、入力画像のピクセル値の範囲を正規化. トレーニングが簡単になる.
        prewhitened = facenet.prewhiten(image)
        prewhitened = prewhitened.reshape(
            -1, prewhitened.shape[0], prewhitened.shape[1], prewhitened.shape[2])
        feed_dict = {self.images_placeholder: prewhitened,
                     self.phase_train_placeholder: False}
        embeddings = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return embeddings


FACE_MEDEL_PATH = './src/models/20180402-114759/20180402-114759.pb'  # 変更
face_embedding = FaceEmbedding(FACE_MEDEL_PATH)

faces_image_paths = glob.glob('./input/*.jpg')

# 顔画像から特徴ベクトルを抽出
features = np.array([face_embedding.face_embeddings(f)[0]
                     for f in faces_image_paths])


# %%
# 2次元に次元削減
PCA
pca = PCA(n_components=2)
pca.fit(features)
reduced = pca.fit_transform(features)

# %%
# クラスタリングラベルを出力
K = 4
kmeans = KMeans(n_clusters=K).fit(reduced)
pred_label = kmeans.predict(reduced)
print(pred_label)


# %%
%matplotlib inline
# クラスタリングした結果をプロット
x = reduced[:, 0]
y = reduced[:, 1]
print(reduced)
print(reduced.shape)
print(x)
print(y)

plt.scatter(x, y, c=pred_label)
plt.colorbar()
plt.show()

# %%


def imscatter(x, y, image_path, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    artists = []
    for x0, y0, image in zip(x, y, image_path):
        image = plt.imread(image)
        im = OffsetImage(image, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    return artists


x = reduced[:, 0]
y = reduced[:, 1]
fig, ax = plt.subplots()
imscatter(x, y, faces_image_paths, ax=ax,  zoom=.2)
ax.plot(x, y, 'ko', alpha=0)
ax.autoscale()
plt.show()


# 追記
# %%
# kmeans.cluster_centers_の返り値（クラスタの重心）とラベルの対応関係を調べる
# 0,1,2の順番であることがわかる
# クラスタの重心を取得
center_of_mass_array = kmeans.cluster_centers_
for i in range(K):
    print(np.mean(center_of_mass_array[i]))

# クラスタごとのベクトルの平均を算出
for i in range(K):
    vec_list = [vec for label, vec in zip(pred_label, reduced) if label == i]
    vec_mean = np.mean(vec_list)
    print(vec_mean)
# %%
# 重心同士の距離を算出


def get_distance(vec1, vec2):
    diff = vec2 - vec1
    distance = np.linalg.norm(diff)
    return distance


distance_map = {}
for i in range(K):
    for j in range(K):
        # 同じクラスタ同士の場合
        if j == i:
            continue
        # 既に計算済みの組み合わせ
        if j < i:
            continue

        distance = get_distance(
            center_of_mass_array[i], center_of_mass_array[j])
        distance_map[f'{i}_to_{j}'] = distance


print(distance_map)
print('\n')
for k, v in distance_map.items():
    print(k)
    print(v)
    print('\n')


# %%
# サンプリング画像をそれぞれのディレクトリにコピー
input_dir = f'./input'
output_dir = f'./output'

for cluster_num in range(K):
    # クラスタ数だけ出力ディレクトリを作成(0, 1, 2, ... K-1)
    cluster_dir = f'{output_dir}/{cluster_num}'
    os.makedirs(cluster_dir, exist_ok=True)

    # 画像のラベルとcluster_numが一致したものを上で作ったディレクトリにコピー
    for img, label in zip(faces_image_paths, pred_label):
        if label == cluster_num:
            cmd = f'cp {img} {cluster_dir}'
            subprocess.run(cmd.split())
