import os
os.environ['PATH'] = '../caffe/build/tools:'+os.environ['PATH']
import sys
sys.path = ['../caffe/python'] + sys.path

import dec
import numpy as np
import caffe
import scipy.io
import dec

def main(db, params):
    n_layer = params['n_layer'][0]
    drop = params['drop'][0]
    encoder_layers = [ ('data', ('data','label',db+'_train', db+'_test', 1.0)) ]
    decoder_layers = [ ('euclid', ('pt_loss', 'd_data', 'data')) ]
    last_dim = params['dim'][0]
    niter = params['iter'][0]
    rate = params['rate'][0]
    for i in xrange(n_layer):
        str_h1 = 'inner%d'%(i+1)
        str_h2 = 'd_inner%d'%(i+1)
        str_x = 'inner%d'%(i)
        str_y = 'd_inner%d'%(i)
        dim = params['dim'][i+1]
        if i == 0:
            str_x = 'data'
            str_y = 'd_data'
        if i == n_layer-1:
            str_h1 = 'output'
            str_h2 = 'output'
        if i != n_layer-1:
            encoder_layers.extend([
                ('inner_init', (str_h1, str_x, dim, np.sqrt(1.0/last_dim))),
                ('relu', (str_h1,)),
                ('drop', (str_h1, drop)),
                ])
        else:
            encoder_layers.extend([
                ('inner_init', (str_h1, str_x, dim, np.sqrt(1.0/last_dim))),
                ('drop', (str_h1, drop)),
                ])
        if i != 0:
            decoder_layers.append(('drop', (str_y, drop)))
            decoder_layers.extend([
                ('relu', (str_y,)),
                ('inner_init', (str_y, str_h2, last_dim, np.sqrt(1.0/dim)))            
                ])
        else:
            decoder_layers.extend([
                ('inner_init', (str_y, str_h2, last_dim, np.sqrt(1.0/dim)))
                           
                ])
        last_dim = dim
    with open('pt_net.prototxt', 'w') as fnet:
        dec.make_net(fnet, encoder_layers+decoder_layers[::-1])

    with open('ft_solver.prototxt', 'w') as fsolver:
        fsolver.write("""net: "pt_net.prototxt"
base_lr: {0}
lr_policy: "step"
gamma: 0.1
stepsize: {1}
display: 1000
test_iter: 100
test_interval: 10000
max_iter: {2}
momentum: 0.9
momentum_burnin: 1000
weight_decay: {3}
snapshot: 10000
snapshot_prefix: "exp/{4}/save"
snapshot_after_train:true
solver_mode: GPU
debug_info: false 
device_id: 0""".format(rate, params['step'][0], niter, params['decay'][0],db ))

    

def pretrain_main(db, params):
    dim = params['dim']
    n_layer = len(dim)-1

    w_down = []
    b_down = []
    for i in xrange(n_layer):
        rate = params['rate'][0]
        layers = [ ('data', ('data','label', db+'_train', db+'_test', 1.0)) ]
        str_x = 'data'
        for j in xrange(i):
            str_h = 'inner%d'%(j+1)
            layers.extend([
                    ('inner_lr', (str_h, str_x, dim[j+1], 0.05, 0.0, 0.0)),
                    ('relu', (str_h,)),
                ])
            str_x = str_h
        if i == n_layer-1:
            str_h = 'output'
        else:
            str_h = 'inner%d'%(i+1)
        if i != 0:
            layers.extend([
                        ('drop_copy', (str_x+'_drop', str_x, params['drop'][0])),
                        ('inner_init', (str_h, str_x+'_drop', dim[i+1], 0.01)),
                ])
        else:
            layers.extend([
                        ('drop_copy', (str_x+'_drop', str_x, 0.0)),
                        ('inner_init', (str_h, str_x+'_drop', dim[i+1], 0.01)),
                ])
        if i != n_layer-1:
            layers.append(('relu', (str_h,)))
            layers.append(('drop', (str_h, params['drop'][0])))
        layers.append(('inner_init', ('d_'+str_x, str_h, dim[i], 0.01)))
        if i != 0:
            layers.append(('relu', ('d_'+str_x,)))
            layers.append(('euclid', ('pt_loss%d'%(i+1), 'd_'+str_x, str_x)))
        else:
            layers.append(('euclid', ('pt_loss%d'%(i+1), 'd_'+str_x, str_x)))

        with open('stack_net.prototxt', 'w') as fnet:
            dec.make_net(fnet, layers)

        with open('pt_solver.prototxt', 'w') as fsolver:
            fsolver.write("""net: "stack_net.prototxt"
base_lr: {0}
lr_policy: "step"
gamma: 0.1
stepsize: {1}
display: 1000
test_iter: 100
test_interval: 10000
max_iter: {2}
momentum: 0.9
momentum_burnin: 1000
weight_decay: {3}
snapshot: 10000
snapshot_prefix: "exp/{4}/save"
snapshot_after_train:true
solver_mode: GPU
debug_info: false 
device_id: 0""".format(rate, params['step'][0], params['pt_iter'][0], params['decay'][0], db))

        if i > 0:
            model = 'exp/'+db+'/save_iter_%d.caffemodel'%params['pt_iter'][0]
        else:
            model = None

        mean, net = dec.extract_feature('stack_net.prototxt', model, 
                                        [str_x], 1, train=True, device=0)

        net.save('stack_init.caffemodel')

        os.system('caffe train --solver=pt_solver.prototxt --weights=stack_init.caffemodel')


        net = caffe.Net('stack_net.prototxt', 'exp/'+db+'/save_iter_%d.caffemodel'%params['pt_iter'][0])
        w_down.append(net.params['d_'+str_x][0].data.copy())
        b_down.append(net.params['d_'+str_x][1].data.copy())
        del net

    net = caffe.Net('pt_net.prototxt', 'exp/'+db+'/save_iter_%d.caffemodel'%params['pt_iter'][0])
    for i in xrange(n_layer):
        if i == 0:
            k = 'd_data'
        else:
            k = 'd_inner%d'%i
        net.params[k][0].data[...] = w_down[i]
        net.params[k][1].data[...] = b_down[i]
    net.save('stack_init_final.caffemodel')





if __name__ == '__main__':
    db = 'mnist'
    input_dim = 784
    #dec.make_mnist_data()
    print main(db, {'n_layer':[4], 'dim': [input_dim, 500, 500, 2000, 10],
               'drop': [0.0], 'rate': [0.1], 'step': [20000], 'iter':[100000], 'decay': [0.0000]})
    print pretrain_main(db, {'dim': [input_dim, 500, 500, 2000, 10], 'pt_iter': [50000],
              'drop': [0.2], 'rate': [0.1], 'step': [20000], 'iter':[100000], 'decay': [0.0000]})
    os.system("caffe train --solver=ft_solver.prototxt --weights=stack_init_final.caffemodel") 

