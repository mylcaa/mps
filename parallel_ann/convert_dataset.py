import idx2numpy
import numpy as np
# arr is now a np.ndarray type of object of shape 60000, 28, 28
file = '/home/koshek/Downloads/archive/train-images-idx3-ubyte/train-images-idx3-ubyte'
arr_images = idx2numpy.convert_from_file(file)

file = '/home/koshek/Downloads/archive/train-labels-idx1-ubyte/train-labels-idx1-ubyte'
arr_labels = idx2numpy.convert_from_file(file)

file = '/home/koshek/Downloads/archive/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
arr_test_images = idx2numpy.convert_from_file(file)

file = '/home/koshek/Downloads/archive/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
arr_test_labels = idx2numpy.convert_from_file(file)

def numbers_to_strings(argument):
    switcher = {
        0: "1 0 0 0 0 0 0 0 0 0 \n",
        1: "0 1 0 0 0 0 0 0 0 0 \n",
        2: "0 0 1 0 0 0 0 0 0 0 \n",
        3: "0 0 0 1 0 0 0 0 0 0 \n",
        4: "0 0 0 0 1 0 0 0 0 0 \n",
        5: "0 0 0 0 0 1 0 0 0 0 \n",
        6: "0 0 0 0 0 0 1 0 0 0 \n",
        7: "0 0 0 0 0 0 0 1 0 0 \n",
        8: "0 0 0 0 0 0 0 0 1 0 \n",
        9: "0 0 0 0 0 0 0 0 0 1 \n",
    }

    return switcher.get(argument, "nothing")

def matrix_to_txt(array_images, array_labels, directory):


    for i in range(0, array_images.shape[0]):
        name_txt = directory
        name_txt = name_txt + str(i) + ".txt"
        f = open(name_txt, "w")
        #take the label and write in the file in the onehot fashion
        f.write(numbers_to_strings(array_labels[i]))
        #write the input image into the file
        for j in range(0, array_images.shape[1], 1):
            for k in range(0, array_images.shape[2], 1):
                f.write('%s ' % array_images[i][j][k]) 
            f.write('\n') 
        f.close()

matrix_to_txt(arr_test_images, arr_test_labels, "/home/koshek/Desktop/MS/zadaci/zad1/test/test_")