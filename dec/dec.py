import os
os.environ['PATH'] = '../caffe/build/tools:'+os.environ['PATH']
import sys
sys.path = ['../caffe/python'] + sys.path

import cv2
import cv
import numpy as np
import shutil
import random
import leveldb
import caffe
from google import protobuf
from caffe.proto import caffe_pb2
from xml.dom import minidom
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist
import cPickle
import time

def vis_square(fname, data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])

    data = data.mean(axis = -1)
    
    plt.imshow(data) 
    plt.savefig(fname)

def vis_cluster(dist, patch_dims, ntop, img):
  cluster = [ [] for i in xrange(dist.shape[1]) ]
  for i in xrange(dist.shape[0]):
    for j in xrange(dist.shape[1]):
      cluster[j].append((i, dist[i,j]))

  cluster.sort(key = lambda x: len(x), reverse = True)
  for i in cluster:
    print len(i)
    i.sort(key = lambda x: x[1], reverse=True)
  viz = np.zeros((patch_dims[0]*len(cluster), patch_dims[1]*ntop, img.shape[-1]))

  for i in xrange(len(cluster)):
    for j in xrange(min(ntop, len(cluster[i]))):
      viz[i*patch_dims[0]:(i+1)*patch_dims[0], j*patch_dims[1]:(j+1)*patch_dims[1], :] = img[cluster[i][j][0]]

  cv2.imwrite('viz_cluster.jpg', viz)

def cluster_acc(Y_pred, Y):
  from sklearn.utils.linear_assignment_ import linear_assignment
  assert Y_pred.size == Y.size
  D = max(Y_pred.max(), Y.max())+1
  w = np.zeros((D,D), dtype=np.int64)
  for i in xrange(Y_pred.size):
    w[Y_pred[i], Y[i]] += 1
  ind = linear_assignment(w.max() - w)
  return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w

def vis_gradient(X, tmm, img):
  from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox

  with open('tmp.pkl') as fin:
    X, tmm, img = cPickle.load(fin)
  img = np.tile(img, 3)

  l = []
  q = tmm.transform(X)
  ind = np.bincount(q.argmax(axis=1)).argmin()
  l = [ i for i in xrange(X.shape[0]) if q[i].argmax() == ind ]
  X = X[l,:]
  img = img[l]

  q = tmm.transform(X)
  q = (q.T/q.sum(axis=1)).T
  p = (q**2)
  p = (p.T/p.sum(axis=1)).T
  grad = 2.0/(1.0+cdist(X, tmm.cluster_centers_, 'sqeuclidean'))*(p-q)*cdist(X, tmm.cluster_centers_, 'cityblock')


  fig, ax = plt.subplots()
  ax.scatter(q[:,ind], grad[:,ind], marker=u'+')

  n_disp = 10
  arg = np.argsort(q[:,ind])
  for i in xrange(n_disp):
    j = arg[int(X.shape[0]*(1.0-1.0*i/n_disp))-1]
    imgbox = OffsetImage(img[j], zoom=1.8)
    ab = AnnotationBbox(imgbox, (q[j,ind], grad[j,ind]),
                        xybox=(0.95-1.0*i/n_disp, 1.06 ),
                        xycoords='data',
                        boxcoords=("axes fraction", "axes fraction"),
                        pad=0.0,
                        arrowprops=dict(arrowstyle="->"))
    ax.add_artist(ab)

  plt.xlabel(r'$q_{ij}$', fontsize=24)
  plt.ylabel(r'$|\frac{\partial L}{\partial z_i}|$', fontsize=24)

  plt.draw()
  plt.show()

def dispImg(X, n, fname=None):
  h = X.shape[1]
  w = X.shape[2]
  c = X.shape[3]
  buff = np.zeros((n*w, n*w, c), dtype=np.uint8)

  for i in xrange(n):
    for j in xrange(n):
      buff[i*h:(i+1)*h, j*w:(j+1)*w, :] = X[i*n+j]

  if fname is None:
    cv2.imshow('a', buff)
    cv2.waitKey(0)
  else:
    cv2.imwrite(fname, buff)

def make_net(fnet, layers):
    layer_dict = {}
    layer_dict['data'] = """layers {{
    name: "{0}"
    type: DATA
    top: "{0}"
    data_param {{
        source: "{2}"
        backend: LEVELDB
        batch_size: 256
    }}
    transform_param {{
        scale: {4}
    }}
    include: {{ phase: TRAIN }}
}}
layers {{
    name: "{0}"
    type: DATA
    top: "{0}"
    data_param {{
        source: "{3}"
        backend: LEVELDB
        batch_size: 100
    }}
    transform_param {{
        scale: {4}
    }}
    include: {{ phase: TEST }}
}}
"""
    layer_dict['data_seek'] = """layers {{
    name: "{0}"
    type: DATA
    top: "{0}"
    data_param {{
        seek: {5}
        source: "{2}"
        backend: LEVELDB
        batch_size: 256
    }}
    transform_param {{
        scale: {4}
    }}
    include: {{ phase: TRAIN }}
}}
layers {{
    name: "{0}"
    type: DATA
    top: "{0}"
    data_param {{
        seek: {5}
        source: "{3}"
        backend: LEVELDB
        batch_size: 100
    }}
    transform_param {{
        scale: {4}
    }}
    include: {{ phase: TEST }}
}}
"""
    layer_dict['sil'] = """layers {{
  name: "{0}silence"
  type: SILENCE
  bottom: "{0}"
}}
"""
    layer_dict['tloss'] = """layers {{
  name: "{0}"
  type: MULTI_T_LOSS
  bottom: "{1}"
  bottom: "{2}"
  blobs_lr: 1.
  blobs_lr: 0.
  blobs_lr: 0.
  top: "loss"
  top: "std"
  top: "ind"
  top: "proba"
  multi_t_loss_param {{
    num_center: {3}
    alpha: 1
    lambda: 2
    beta: 1
    bandwidth: 0.1
    weight_filler {{
      type: 'gaussian'
      std: 0.5
    }}
  }}
}}
layers {{
  name: "silence"
  type: SILENCE
  bottom: "label"
  bottom: "ind"
  bottom: "proba"
}}
"""
    layer_dict['inner'] = """layers {{
  name: "{0}"
  type: INNER_PRODUCT
  bottom: "{1}"
  top: "{0}"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: {2} 
    weight_filler {{
      type: "gaussian"
      std: 0.05
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}
"""
    layer_dict['inner_init'] = """layers {{
  name: "{0}"
  type: INNER_PRODUCT
  bottom: "{1}"
  top: "{0}"
  blobs_lr: 1
  blobs_lr: 2
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: {2} 
    weight_filler {{
      type: "gaussian"
      std: {3}
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}
"""
    layer_dict['inner_lr'] = """layers {{
  name: "{0}"
  type: INNER_PRODUCT
  bottom: "{1}"
  top: "{0}"
  blobs_lr: {4}
  blobs_lr: {5}
  weight_decay: 1
  weight_decay: 0
  inner_product_param {{
    num_output: {2} 
    weight_filler {{
      type: "gaussian"
      std: {3}
    }}
    bias_filler {{
      type: "constant"
      value: 0
    }}
  }}
}}
"""
    layer_dict['relu'] = """layers {{
  name: "{0}relu"
  type: RELU
  bottom: "{0}"
  top: "{0}"
}}
"""
    layer_dict['drop'] = """layers {{
  name: "{0}drop"
  type: DROPOUT
  bottom: "{0}"
  top: "{0}"
  dropout_param {{
    dropout_ratio: {1}
  }}
}}
"""
    layer_dict['drop_copy'] = """layers {{
  name: "{0}drop"
  type: DROPOUT
  bottom: "{1}"
  top: "{0}"
  dropout_param {{
    dropout_ratio: {2}
  }}
}}
"""
    layer_dict['euclid'] = """layers {{
  name: "{0}"
  type: EUCLIDEAN_LOSS
  bottom: "{1}"
  bottom: "{2}"
  top: "{0}"
}}
"""

    fnet.write('name: "net"\n')
    for k,v in layers:
        fnet.write(layer_dict[k].format(*v))
    fnet.close()

class TMM(object):
  def __init__(self, n_components=1, alpha=1):
    self.n_components = n_components
    self.tol = 1e-5
    self.alpha = float(alpha)

  def fit(self, X):
    from sklearn.cluster import KMeans
    kmeans = KMeans(self.n_components, n_init=20)
    kmeans.fit(X)
    self.cluster_centers_ = kmeans.cluster_centers_
    self.covars_ = np.ones(self.cluster_centers_.shape)

  def transform(self, X):
    p = 1.0
    dist = cdist(X, self.cluster_centers_)
    r = 1.0/(1.0+dist**2/self.alpha)**((self.alpha+p)/2.0)
    r = (r.T/r.sum(axis=1)).T
    return r

  def predict(self, X):
    return self.transform(X).argmax(axis=1)

def load_mnist(root, training):
  if training:
    data = 'train-images-idx3-ubyte'
    label = 'train-labels-idx1-ubyte'
    N = 60000
  else:
    data = 't10k-images-idx3-ubyte'
    label = 't10k-labels-idx1-ubyte'
    N = 10000
  with open(root+data, 'rb') as fin:
    fin.seek(16, os.SEEK_SET)
    X = np.fromfile(fin, dtype=np.uint8).reshape((N,28*28))
  with open(root+label, 'rb') as fin:
    fin.seek(8, os.SEEK_SET)
    Y = np.fromfile(fin, dtype=np.uint8)
  return X, Y

def make_mnist_data():
  X, Y = load_mnist('../mnist/', True)
  X = X.astype(np.float64)*0.02
  write_db(X, Y, 'mnist_train')

  X_, Y_ = read_db('mnist_train', True)
  assert np.abs((X - X_)).mean() < 1e-5
  assert (Y != Y_).sum() == 0

  X2, Y2 = load_mnist('../mnist/', False)
  X2 = X2.astype(np.float64)*0.02
  write_db(X2, Y2, 'mnist_test')

  X3 = np.concatenate((X,X2), axis=0)
  Y3 = np.concatenate((Y,Y2), axis=0)
  write_db(X3,Y3, 'mnist_total')

def make_reuters_data():
  np.random.seed(1234)
  random.seed(1234)
  from sklearn.feature_extraction.text import CountVectorizer
  did_to_cat = {}
  cat_list = ['CCAT', 'GCAT', 'MCAT', 'ECAT']
  with open('../reuters/rcv1-v2.topics.qrels') as fin:
    for line in fin.readlines():
      line = line.strip().split(' ')
      cat = line[0]
      did = int(line[1])
      if cat in cat_list:
        did_to_cat[did] = did_to_cat.get(did, []) + [cat]
    for did in did_to_cat.keys():
      if len(did_to_cat[did]) > 1:
        del did_to_cat[did]

  dat_list = ['lyrl2004_tokens_test_pt0.dat', 
              'lyrl2004_tokens_test_pt1.dat',
              'lyrl2004_tokens_test_pt2.dat',
              'lyrl2004_tokens_test_pt3.dat',
              'lyrl2004_tokens_train.dat']
  data = []
  target = []
  cat_to_cid = {'CCAT':0, 'GCAT':1, 'MCAT':2, 'ECAT':3}
  del did
  for dat in dat_list:
    with open('../reuters/'+dat) as fin:
      for line in fin.readlines():
        if line.startswith('.I'):
          if 'did' in locals():
            assert doc != ''
            if did_to_cat.has_key(did):
              data.append(doc)
              target.append(cat_to_cid[did_to_cat[did][0]])
          did = int(line.strip().split(' ')[1])
          doc = ''
        elif line.startswith('.W'):
          assert doc == ''
        else:
          doc += line

  assert len(data) == len(did_to_cat)

  X = CountVectorizer(dtype=np.float64, max_features=2000).fit_transform(data)
  Y = np.asarray(target)

  from sklearn.feature_extraction.text import TfidfTransformer
  X = TfidfTransformer(norm='l2', sublinear_tf=True).fit_transform(X)
  X = np.asarray(X.todense())*np.sqrt(X.shape[1])

  p = np.random.permutation(X.shape[0])
  X = X[p]
  Y = Y[p]

  N = X.shape[0]
  write_db(X[:N], Y[:N], 'reutersidf_train')
  write_db(X[N*4/5:N], Y[N*4/5:N], 'reutersidf_test')
  write_db(X[:N], Y[:N], 'reutersidf_total')
  np.save('reutersidf.npy', Y[:N])

  N = 10000
  write_db(X[:N], Y[:N], 'reutersidf10k_train')
  write_db(X[N*4/5:N], Y[N*4/5:N], 'reutersidf10k_test')
  write_db(X[:N], Y[:N], 'reutersidf10k_total')

def hog_picture(hog, resolution):
    from scipy.misc import imrotate
    glyph1 = np.zeros((resolution, resolution), dtype=np.uint8)
    glyph1[:, round(resolution / 2)-1:round(resolution / 2) + 1] = 255
    glyph = np.zeros((resolution, resolution, 9), dtype=np.uint8)
    glyph[:, :, 0] = glyph1
    for i in xrange(1, 9):
        glyph[:, :, i] = imrotate(glyph1, -i * 20)

    shape = hog.shape
    clamped_hog = hog.copy()
    clamped_hog[hog < 0] = 0
    image = np.zeros((resolution * shape[0], resolution * shape[1]), dtype=np.float32)
    for i in xrange(shape[0]):
        for j in xrange(shape[1]):
            for k in xrange(9):
                image[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution] = np.maximum(image[i*resolution:(i+1)*resolution, j*resolution:(j+1)*resolution], clamped_hog[i, j, k] * glyph[:, :, k])

    return image

def load_stl(fname):
  from joblib import Parallel, delayed
  import features

  X = np.fromfile('../stl/'+fname, dtype=np.uint8)
  X = X.reshape((X.size/3/96/96, 3, 96, 96)).transpose((0,3,2,1))
  dispImg(X[:100, :, :, [2,1,0]], 10, fname+'_org.jpg')

  n_jobs = 10
  cmap_size = (8,8)
  N = X.shape[0]

  H = np.asarray(Parallel(n_jobs=n_jobs)( delayed(features.hog)(X[i]) for i in xrange(N) ))

  H_img = np.repeat(np.asarray([ hog_picture(H[i], 9) for i in xrange(100) ])[:, :,:,np.newaxis], 3, 3)
  dispImg(H_img, 10, fname+'_hog.jpg') 
  H = H.reshape((H.shape[0], H.size/N))

  X_small = np.asarray(Parallel(n_jobs=n_jobs)( delayed(cv2.resize)(X[i], cmap_size) for i in xrange(N) ))
  crcb = np.asarray(Parallel(n_jobs=n_jobs)( delayed(cv2.cvtColor)(X_small[i], cv.CV_RGB2YCrCb) for i in xrange(N) ))
  crcb = crcb[:,:,:,1:]
  crcb = crcb.reshape((crcb.shape[0], crcb.size/N))

  feature = np.concatenate(((H-0.2)*10.0, (crcb-128.0)/10.0), axis=1)
  print feature.shape

  return feature, X[:,:,:,[2,1,0]]

def make_stl_data():
  np.random.seed(1234)
  random.seed(1234)
  X_train, img_train = load_stl('train_X.bin')
  X_test, img_test = load_stl('test_X.bin')
  X_unlabel, img_unlabel = load_stl('unlabeled_X.bin')
  Y_train = np.fromfile('../stl/train_y.bin', dtype=np.uint8) - 1
  Y_test = np.fromfile('../stl/test_y.bin', dtype=np.uint8) - 1

  X_total = np.concatenate((X_train, X_test), axis=0)
  img_total = np.concatenate((img_train, img_test), axis=0)
  Y_total = np.concatenate((Y_train, Y_test))
  p = np.random.permutation(X_total.shape[0])
  X_total = X_total[p]
  img_total = img_total[p]
  Y_total = Y_total[p]
  write_db(X_total, Y_total, 'stl_total')
  write_db(img_total, Y_total, 'stl_img')

  X = np.concatenate((X_total, X_unlabel), axis=0)
  p = np.random.permutation(X.shape[0])
  X = X[p]
  Y = np.zeros((X.shape[0],))
  N = X.shape[0]*4/5
  write_db(X[:N], Y[:N], 'stl_train')
  write_db(X[N:], Y[N:], 'stl_test')


def read_db(str_db, float_data = True):
    db = leveldb.LevelDB(str_db)
    datum = caffe_pb2.Datum()
    array = []
    label = []
    for k,v in db.RangeIter():
        dt = datum.FromString(v)
        if float_data:
          array.append(dt.float_data)
        else: 
          array.append(np.fromstring(dt.data, dtype=np.uint8))
        label.append(dt.label)
    return np.asarray(array), np.asarray(label)

def write_db(X, Y, fname):
    if os.path.exists(fname):
      shutil.rmtree(fname)
    assert X.shape[0] == Y.shape[0]
    X = X.reshape((X.shape[0], X.size/X.shape[0], 1, 1))
    db = leveldb.LevelDB(fname)

    for i in xrange(X.shape[0]):
      x = X[i]
      if x.ndim != 3:
        x = x.reshape((x.size,1,1))
      db.Put('{:08}'.format(i), caffe.io.array_to_datum(x, int(Y[i])).SerializeToString())
    del db

def update_db(seek, N, X, Y, fname):
    assert X.shape[0] == Y.shape[0]
    X = X.reshape((X.shape[0], X.size/X.shape[0], 1, 1))
    db = leveldb.LevelDB(fname)

    for i in xrange(X.shape[0]):
      x = X[i]
      if x.ndim != 3:
        x = x.reshape((x.size,1,1))
      db.Put('{:08}'.format((i+seek)%N), caffe.io.array_to_datum(x, int(Y[i])).SerializeToString())
    del db

def extract_feature(net, model, blobs, N, train = False, device = None):
  if type(net) is str:
    if train:
      caffe.Net.set_phase_train()
    if model:
      net = caffe.Net(net, model)
    else:
      net = caffe.Net(net)
    caffe.Net.set_phase_test()
  if not (device is None):
    caffe.Net.set_mode_gpu()
    caffe.Net.set_device(device)

  batch_size = net.blobs[blobs[0]].num
  res = [ [] for i in blobs ]
  for i in xrange((N-1)/batch_size+1):
    ret = net.forward(blobs=blobs)
    for i in xrange(len(blobs)):
      res[i].append(ret[blobs[i]].copy())

  for i in xrange(len(blobs)):
    res[i] = np.concatenate(res[i], axis=0)[:N]

  return res, net

def write_net(db, dim, n_class, seek):
  layers = [ ('data_seek', ('data','dummy',db+'_total', db+'_total', 1.0, seek)),
             ('data_seek', ('label', 'dummy', 'train_weight', 'train_weight', 1.0, seek)),

             ('inner', ('inner1', 'data', 500)),
             ('relu', ('inner1',)),

             ('inner', ('inner2', 'inner1', 500)),
             ('relu', ('inner2',)),

             ('inner', ('inner3', 'inner2', 2000)),
             ('relu', ('inner3',)),

             ('inner', ('output', 'inner3', dim)),

             ('tloss', ('loss', 'output', 'label', n_class))
          ]
  with open('net.prototxt', 'w') as fnet:
    make_net(fnet, layers)

def DisKmeans(db, update_interval = None):
    from sklearn.cluster import KMeans
    from sklearn.mixture import GMM
    from sklearn.lda import LDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score
    from scipy.spatial.distance import cdist 
    import cPickle
    from scipy.io import loadmat

    if db == 'mnist':
      N_class = 10
      batch_size = 100
      train_batch_size = 256
      X, Y = read_db(db+'_total', True)
      X = np.asarray(X, dtype=np.float64)
      Y = np.asarray(np.squeeze(Y), dtype = np.int32)
      N = X.shape[0]
      img = np.clip((X/0.02), 0, 255).astype(np.uint8).reshape((N, 28, 28, 1))
    elif db == 'stl':
      N_class = 10
      batch_size = 100
      train_batch_size = 256
      img = read_db('stl_img', False)[0]
      img = img.reshape((img.shape[0], 96, 96, 3))
      X, Y = read_db(db+'_total', True)
      X = np.asarray(X, dtype=np.float64)
      Y = np.asarray(np.squeeze(Y), dtype = np.int32)
      N = X.shape[0]
    elif db == 'reuters':
      N_class = 4
      batch_size = 100
      train_batch_size = 256
      Y = np.fromfile('reuters.npy', dtype=np.int64)
      N = Y.shape[0]
    elif db == 'reutersidf':
      N_class = 4
      batch_size = 100
      train_batch_size = 256
      Y = np.load('reutersidf.npy')
      N = Y.shape[0]
    elif db == 'reuters10k' or db == 'reutersidf10k':
      N_class = 4
      batch_size = 100
      train_batch_size = 256
      X, Y = read_db(db+'_total', True)
      X = np.asarray(X, dtype=np.float64)
      Y = np.asarray(np.squeeze(Y), dtype = np.int32)
      N = X.shape[0]

    tmm_alpha = 1.0
    total_iters = (N-1)/train_batch_size+1
    if not update_interval:
      update_interval = total_iters
    Y_pred = np.zeros((Y.shape[0]))
    iters = 0
    seek = 0
    dim = 10


    acc_list = []

    while True:
      write_net(db, dim, N_class, "'{:08}'".format(0))
      if iters == 0:
        write_db(np.zeros((N,N_class)), np.zeros((N,)), 'train_weight')
        ret, net = extract_feature('net.prototxt', 'exp/'+db+'/save_iter_100000.caffemodel', ['output'], N, True, 0)
        feature = ret[0].squeeze()

        gmm_model = TMM(N_class)
        gmm_model.fit(feature)
        net.params['loss'][0].data[0,0,:,:] = gmm_model.cluster_centers_.T
        net.params['loss'][1].data[0,0,:,:] = 1.0/gmm_model.covars_.T
      else:
        ret, net = extract_feature('net.prototxt', 'init.caffemodel', ['output'], N, True, 0)
        feature = ret[0].squeeze()

        gmm_model.cluster_centers_ = net.params['loss'][0].data[0,0,:,:].T


      Y_pred_last = Y_pred
      Y_pred = gmm_model.predict(feature).squeeze()
      acc, freq = cluster_acc(Y_pred, Y)
      acc_list.append(acc)
      nmi = normalized_mutual_info_score(Y, Y_pred)
      print freq
      print freq.sum(axis=1)
      print 'acc: ', acc, 'nmi: ', nmi
      print (Y_pred != Y_pred_last).sum()*1.0/N
      if (Y_pred != Y_pred_last).sum() < 0.001*N:
        print acc_list
        return acc, nmi
      time.sleep(1)

      write_net(db, dim, N_class, "'{:08}'".format(seek))
      weight = gmm_model.transform(feature)

      weight = (weight.T/weight.sum(axis=1)).T
      bias = (1.0/weight.sum(axis=0))
      bias = N_class*bias/bias.sum()
      weight = (weight**2)*bias
      weight = (weight.T/weight.sum(axis=1)).T
      print weight[:10,:]
      write_db(weight, np.zeros((weight.shape[0],)), 'train_weight')

      net.save('init.caffemodel')
      del net 

      with open('solver.prototxt', 'w') as fsolver:
            fsolver.write("""net: "net.prototxt"
base_lr: 0.01
lr_policy: "step"
gamma: 0.1
stepsize: 100000
display: 10
max_iter: %d
momentum: 0.9
weight_decay: 0.0000
snapshot: 100
snapshot_prefix: "exp/test/save"
snapshot_after_train:true
solver_mode: GPU
debug_info: false
sample_print: false
device_id: 0"""%update_interval)
      os.system('caffe train --solver=solver.prototxt --weights=init.caffemodel')
      shutil.copyfile('exp/test/save_iter_%d.caffemodel'%update_interval, 'init.caffemodel')

      iters += 1
      seek = (seek + train_batch_size*update_interval)%N

if __name__ == '__main__':
    db = sys.argv[1]
    if db == 'mnist':
        lam = 160
    elif db == 'stl':
        lam = 40
    elif db == 'reutersidf' or db == 'reutersidf10k':
        lam = 20
    else:
        lam = int(sys.argv[2])
    """acc_list = []
    nmi_list = []
    for i in xrange(0,9):
      lam = 10*(2**i)
      acc, nmi = DisKmeans(db, lam)
      acc_list.append(acc)
      nmi_list.append(nmi)
    print acc_list
    print nmi_list"""
    DisKmeans(db, lam)
    
    
