# K01_IF3270_Tubes1_ML
Tugas Besar 1 Pembelajaran Mesin: Implementasi Feed Forward Neural Network

# How to setup
lakukan cloning pada link github ini 
```bash
git clone https://github.com/bagassambega/K01-IF3270-Tubes1-ML.git
```
pindah ke folder src/ dan lakukan command berikut
```bash
pip install requirements.txt
```
## How to run 
1. run the program in the main.ipynb

```python
ffnn = FFNN(x=temp_x, y=temp_y, x_val=temp_x_val, y_val=temp_y_val, total_layers=[10], loss_function="mse", weight_method="xavier", learning_rate=0.01, activations=["relu", "softmax"], verbose=True, epochs=3, seed=42)
```
kode di atas adalah cuplikan untuk membuat instance dari model 
untuk menambah layer pada model kita bisa menambahkan angka baru pada total_layers 

`total_layers = [10]`: ini maksudnya terdiri dari 1 input layer, 1 hidden layer, dan 1 output layer angka 10 pada list tersebut itu adalah jumlah neuron pada setiap layer

`activations=["relu", "softmax"]`: ini maksudnya adalah input layer ke hidden layer 1 memakai relu dan hidden layer 1 ke output layer memakai softmax





## Pembagian Tugas 
| NIM          | Pembagian Tugas      | 
|---------------|---------------|
| 13522071        |  Struktur FFNN, automatic differentiation, backward propagation, dan Laporan    | 
| 13522097        |  Inisialisasi weight (+ bonus Xavier dan He), one-hot encoding, backward propagation, dan Laporan   | 
| 13522119        |  Forward propagation, loss function, activation function, turunan fungsi loss dan activation,  L1 & L2 Regularization, Save and Load mode, dan Laporan  | 
