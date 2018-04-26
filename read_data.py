import numpy as np
import scipy.misc as misc
import struct
import os

def read_data(path):
  
  with open(path, 'rb') as fin:
    
    buffers = fin.read()
    
    head = struct.unpack_from('>IIII', buffers, 0)

    offset = struct.calcsize('>IIII')
    num_img = head[1]
    width = head[2]
    height = head[3]

    bits = num_img*width*height
    bits_string = '>'+str(bits)+'B'

    data = struct.unpack_from(bits_string, buffers, offset)
    data = np.array(data)
    data = np.reshape(data,(num_img,height,width))

    return data
 
def read_label(path):
  
  with open(path, 'rb') as fin:
    
    buffers = fin.read()

    head = struct.unpack_from('>IIII', buffers, 0)

    num_label = head[1]

    offset = struct.calcsize('>II')

    bits_string = '>'+str(num_label)+'B'

    labels = struct.unpack_from(bits_string, buffers, offset)

    labels = np.array(labels)

    return labels

def read_mnist(mnist_dir):
  
  train_data_path = os.path.join(mnist_dir,'train-images.idx3-ubyte')
  train_label_path = os.path.join(mnist_dir,'train-labels.idx1-ubyte')
  test_data_path = os.path.join(mnist_dir,'t10k-images.idx3-ubyte')
  test_label_path = os.path.join(mnist_dir,'t10k-labels.idx1-ubyte')
  train_data = read_data(train_data_path)
  train_label = read_label(train_label_path)
  test_data = read_data(test_data_path)
  test_label = read_label(test_label_path)

  return train_data, train_label, test_data, test_label
  

if __name__ == '__main__':
  
  read_mnist('./')
