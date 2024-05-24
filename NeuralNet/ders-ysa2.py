import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

data_path = 'C:/___MachineLearningAndAI/YZ/AI_Codes_Lesson/NeuralNet/heart_statlog_cleveland_hungary_final.csv'

datas = pd.read_csv(data_path)



inputs = datas.drop('target', axis=1).values

outputs = datas['target'].values.reshape(-1,1)

data_size = inputs.shape[0]
input_size = inputs.shape[1]  # Bağımsız değişken sayısı
print(data_size, input_size)

#inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
#outputs = np.array([[0,1,1,0]])

data_size = inputs.shape[0]
input_size = inputs.shape[1]

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
inputs = ss.fit_transform(inputs)


layer1_neuron_size = 10 # 1. gizli katman
layer2_neuron_size = 5 # 2. gizli katman
output_neuron_size = 1


weights_1 = np.random.randn(input_size, layer1_neuron_size)
#bias1 = np.random.randn(1, layer1_neuron_size)
bias1 = np.zeros((1, layer1_neuron_size))


weights_2 = np.random.randn(layer1_neuron_size, layer2_neuron_size)
#bias2 = np.random.randn(1, layer2_neuron_size)
bias2 = np.zeros((1, layer2_neuron_size))


output_weights = np.random.randn(layer2_neuron_size, output_neuron_size)
#bias_output = np.random.randn(1, output_neuron_size)
bias_output = np.zeros((1, output_neuron_size))

learning_rate = 0.1
epochs = 100000


for i in range(epochs):
    layer1_net = np.dot(inputs, weights_1) + bias1
    layer1_output = sigmoid(layer1_net)
    
    layer2_net = np.dot(layer1_output, weights_2) + bias2
    layer2_output = sigmoid(layer2_net)
   
    output_net = np.dot(layer2_output, output_weights) + bias_output
    predict_output = sigmoid(output_net)
    
    # hata hesaplama
    error_output = outputs - predict_output 
    
    delta_error_output = error_output*sigmoid_derivative(predict_output)
    
    error_layer2 = (delta_error_output).dot(output_weights.T)
    
    delta_error_layer2 = error_layer2 *sigmoid_derivative(layer2_output)
    
    error_layer1 = delta_error_layer2.dot(weights_2.T)
    
    delta_error_layer1 = error_layer1 *sigmoid_derivative(layer1_output)
   
   
    
    # Geri yayılım
    output_weights += learning_rate * layer2_output.T.dot(delta_error_output)
    bias_output += learning_rate*np.sum(delta_error_output,axis=0, keepdims=True)
    
    
    weights_2 += learning_rate*layer1_output.T.dot(delta_error_layer2)
    bias2  += learning_rate*np.sum(delta_error_layer2,axis=0, keepdims=True)
    
    
    weights_1 += learning_rate*inputs.T.dot(delta_error_layer1)
    bias1 += learning_rate*np.sum(delta_error_layer1,axis=0, keepdims=True)
    
    # Her epoch sonunda hatayı ve doğruluğu yazdırma
    if (i + 1) % 100 == 0 or i == 0:  # Her 100 epoch'da bir veya ilk epoch'da yazdır
       loss = np.mean(np.abs(error_output))
       predictions = (predict_output > 0.5).astype(int)
       accuracy = np.mean(predictions == outputs)
       print(f"Epoch {i+1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy * 100:.2f}%")
    
    
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        