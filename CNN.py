import numpy as np
import matplotlib.pyplot as plt
import time

#First Step of the Project

# A function to plot images
def show_image(img):
    image = img.reshape((28, 28))
    plt.imshow(image, 'gray')

# Reading The Train Set
train_images_file = open('train-images.idx3-ubyte', 'rb')
train_images_file.seek(4)
num_of_train_images = int.from_bytes(train_images_file.read(4), 'big')
#num_of_train_images = 100
train_images_file.seek(16)

train_labels_file = open('train-labels.idx1-ubyte', 'rb')
train_labels_file.seek(8)

train_set = []
for n in range(num_of_train_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i, 0] = int.from_bytes(train_images_file.read(1), 'big') / 256
    
    label_value = int.from_bytes(train_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1
    
    train_set.append((image, label))

# Reading The Test Set
test_images_file = open('t10k-images.idx3-ubyte', 'rb')
test_images_file.seek(4)

test_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
test_labels_file.seek(8)

num_of_test_images = int.from_bytes(test_images_file.read(4), 'big')
test_images_file.seek(16)

test_set = []
for n in range(num_of_test_images):
    image = np.zeros((784, 1))
    for i in range(784):
        image[i] = int.from_bytes(test_images_file.read(1), 'big') / 256

    label_value = int.from_bytes(test_labels_file.read(1), 'big')
    label = np.zeros((10, 1))
    label[label_value, 0] = 1

    test_set.append((image, label))

# Plotting an image
show_image(train_set[0][0])
print(train_set[0][1])
plt.show()

#Second Step of the Project

#Activation Function = Sigmoid
def sig(x):
    return 1/(1 + np.exp(-x))

#Weights with normal random numbers and zero biases
first_layer_weight = np.random.normal(size=(16,784))
second_layer_weight = np.random.normal(size=(16,16))
third_layer_weight = np.random.normal(size=(10,16))
first_layer_bias = np.zeros((16, 1))
second_layer_bias = np.zeros((16,1))
third_layer_bias = np.zeros((10,1))

batch_size = 50
learning_rate = 1
number_of_epochs = 5
number_of_samples = 100
epoch_cost = []

#output of the network for 100 inputs
correct_output_step2 = 0
#Calculate output of the network for 100 inputs and check the accuracy
for i in range(100):
    nn_inputs = train_set[i][0]
    first_Layer_output = sig(first_layer_weight @ nn_inputs + first_layer_bias)
    second_layer_output = sig(second_layer_weight @ first_Layer_output + second_layer_bias)
    nn_outputs = sig(third_layer_weight @ second_layer_output + third_layer_bias)
    activated_output = np.where(nn_outputs == np.amax(nn_outputs))[0][0]
    if activated_output == np.where(train_set[i][1] == 1)[0][0]:
        correct_output_step2 += 1

print("The Accuracy of the Neural Network for the first 100 images in training set is: ", correct_output_step2/100)



#Third Step of the Project

#forth step
start = time.time()
for i in range(number_of_epochs):
    #np.random.shuffle(train_set)
    mini_batches = [train_set[x:x + batch_size] for x in range(0, num_of_train_images, batch_size)]
    for mb in mini_batches:
        #initialize grad_W and grad_b to zero for each batch
        first_grad_W = np.zeros((16, 784))
        second_grad_W = np.zeros((16, 16))
        third_grad_W = np.zeros((10, 16))
        first_grad_b = np.zeros((16, 1))
        second_grad_b = np.zeros((16, 1))
        third_grad_b = np.zeros((10, 1))
        for image in mb:
            #comput the output of neural network for each image
            nn_outputs_star = image[1]
            nn_inputs = image[0]
            first_Layer_output = sig(first_layer_weight @ nn_inputs + first_layer_bias)
            second_layer_output = sig(second_layer_weight @ first_Layer_output + second_layer_bias)
            nn_outputs = sig(third_layer_weight @ second_layer_output + third_layer_bias)
            d_cost_d_second_layer_output = np.zeros(16)
            d_cost_d_first_layer_output = np.zeros(16)
            #last layer
            #update third_grad_W (dcost/d(third_layer_weight)) and third_grad_b (dcost/d(third_layer_bias))
            third_grad_W += 2*(nn_outputs-nn_outputs_star)*(nn_outputs*(1-nn_outputs))@np.transpose(second_layer_output)
            third_grad_b += 2*(nn_outputs-nn_outputs_star)*(nn_outputs*(1-nn_outputs))
            #calculate dcost/d(second_layer_output)
            d_cost_d_second_layer_output = (np.transpose(third_layer_weight))@((2*(nn_outputs-nn_outputs_star))*(nn_outputs)*(1-nn_outputs))
            #second layer
            # update second_grad_W (dcost/d(second_layer_weight)) and second_grad_b (dcost/d(second_layer_bias))
            second_grad_W += d_cost_d_second_layer_output*(second_layer_output*(1-second_layer_output))@np.transpose(first_Layer_output)
            second_grad_b += d_cost_d_second_layer_output*(second_layer_output*(1-second_layer_output))
            d_cost_d_first_layer_output = np.transpose(second_layer_weight)@(d_cost_d_second_layer_output*(second_layer_output*(1-second_layer_output)))
            #firs layer
            first_grad_W += d_cost_d_first_layer_output*(first_Layer_output*(1-first_Layer_output))@np.transpose(nn_inputs)
            first_grad_b += d_cost_d_first_layer_output*(first_Layer_output*(1-first_Layer_output))

        first_layer_weight = first_layer_weight - (learning_rate*(first_grad_W/batch_size))
        second_layer_weight = second_layer_weight - (learning_rate*(second_grad_W/batch_size))
        third_layer_weight = third_layer_weight - (learning_rate*(third_grad_W/batch_size))
        first_layer_bias = first_layer_bias - (learning_rate*(first_grad_b/batch_size))
        second_layer_bias = second_layer_bias - (learning_rate*(second_grad_b/batch_size))
        third_layer_bias = third_layer_bias - (learning_rate*(third_grad_b/batch_size))

    cost = 0
    for image in train_set:
        nn_outputs_star = image[1]
        nn_inputs = image[0]
        first_Layer_output = sig(first_layer_weight @ nn_inputs + first_layer_bias)
        second_layer_output = sig(second_layer_weight @ first_Layer_output + second_layer_bias)
        nn_outputs = sig(third_layer_weight @ second_layer_output + third_layer_bias)
        for j in range(10):
            cost += (nn_outputs[j, 0] - nn_outputs_star[j, 0]) ** 2
    cost /= 100
    epoch_cost.append(cost)
end = time.time()
#check the accuracy of nn with learning for train set for step 4
correct_output_step4 = 0
for i in range(100):
    nn_inputs = train_set[i][0]
    first_Layer_output = sig(first_layer_weight @ nn_inputs + first_layer_bias)
    second_layer_output = sig(second_layer_weight @ first_Layer_output + second_layer_bias)
    nn_outputs = sig(third_layer_weight @ second_layer_output + third_layer_bias)
    activated_output = np.where(nn_outputs == np.amax(nn_outputs))[0][0]
    if activated_output == np.where(train_set[i][1] == 1)[0][0]:
        correct_output_step4 += 1

#check the accuracy of nn with learning for train set for step 5
correct_output_step4 = 0
for i in range(num_of_train_images):
    nn_inputs = train_set[i][0]
    first_Layer_output = sig(first_layer_weight @ nn_inputs + first_layer_bias)
    second_layer_output = sig(second_layer_weight @ first_Layer_output + second_layer_bias)
    nn_outputs = sig(third_layer_weight @ second_layer_output + third_layer_bias)
    activated_output = np.where(nn_outputs == np.amax(nn_outputs))[0][0]
    if activated_output == np.where(train_set[i][1] == 1)[0][0]:
        correct_output_step4 += 1

#check the accuracy of nn with learning for test set
correct_output_test = 0
for i in range(num_of_test_images):
    nn_inputs = test_set[i][0]
    first_Layer_output = sig(first_layer_weight @ nn_inputs + first_layer_bias)
    second_layer_output = sig(second_layer_weight @ first_Layer_output + second_layer_bias)
    nn_outputs = sig(third_layer_weight @ second_layer_output + third_layer_bias)
    activated_output = np.where(nn_outputs == np.amax(nn_outputs))[0][0]
    if activated_output == np.where(test_set[i][1] == 1)[0][0]:
        correct_output_test += 1


#print the result
t = (end-start)/60
print("The Accuracy of the Neural Network Model in train set is: ", correct_output_step4/num_of_train_images)
print("The Accuracy of the Neural Network Model in test set is: ", correct_output_test/num_of_test_images)
print("Duration of learning (as a minutes): ", end="")
print ("{0:.2f}".format(t))

plt.plot([i for i in range(number_of_epochs)], epoch_cost)
plt.show()