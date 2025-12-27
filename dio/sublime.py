"""Ferramenta: Sublime text - interface que foi utilizada nas aulas
ferramenta: replit em nuvem: programação no navegador

Tipos de variáveis para trabalhar com ML:
1. tipo int (inteiro)
2. ponto flutuante ou decimal (float)
3. complexo (complex) representar números complexos, cálculos geométricos e científicos
4. String (str) - conjunto de caracteres
5. booleano (bool) - dado lógico, pode assumir só dois valores: falso ou verdadeiro

-- Estrutura condicional simples
if soma > 0:
    print ("maior que zero")

-- Estrutura condicional aninhada
if soma > 0:
    print("soma maior do que zero")
elif soma = 0:
    print("soma é igual a zero")
else: 
    print("soma é menor do que zero")

-- Estruturas de repetição
1. for:
lista = [1,2,3,4]
for item in lista:
    print(item)

2. while:
contador = 0
while contador < 10:
    print(f'valor do contador é: {contador}")
    contador +=1

-- Matplotlib
documentação:https://matplotlib.org/stable/index.html
Primeiro exemplo: 
"""
import matplotlib
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
x,y = make_regression(n_samples=200, n_features=1,noise=30)

plt.scatter(x,y)
plt.show()

#tensorflow playground
#mnist 

#rede neural do zero
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.ToTensor()

trainset = datasets.MNIST('./MNIST_data/', download = True, train=True,transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = 64, suffle=True)

valset=datasets.MNIST('./MNIST_data/', download = True, train=True,transform=transform)
valloader=torch.utils.data.DataLoader(trainset,batch_size = 64, suffle=True)

dataiter = iter(trainloader)
imagens, etiquetas = dataiter.next()
plt.imshow(imagens[0].numpy().squeeze(), cmap='gray_r');

print(imagens[0].shape)#para verificar as dimensões do tensor de cada imagem
print(etiquetas[0].shape)#para verificar a dimensão do tensor de cada etiqueta

#keras inception -> keras applications
#copiar as camadas da rede inception
""" 
class Modelo(nn.Module):
    def __init
 """