import os
import dec
print 'Building HOG feature extractor...'
os.system('python setup_features.py build')
os.system('python setup_features.py install')

print 'Preparing stl data. This could take a while...'
dec.make_stl_data()
