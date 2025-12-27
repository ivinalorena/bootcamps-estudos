# Classificação de Imagens com CNN e Transfer Learning (VGG16)
---
### Descrição do Projeto

Este projeto tem como objetivo desenvolver e comparar modelos de classificação de imagens utilizando Redes Neurais Convolucionais (CNNs) treinadas do zero e com Transfer Learning, empregando a arquitetura VGG16 pré-treinada no conjunto de dados ImageNet.

O estudo demonstra como o uso de modelos pré-treinados pode melhorar significativamente o desempenho quando se trabalha com conjuntos de dados relativamente pequenos.

### Dataset

Nome: 101 Object Categories (Caltech-101)

Quantidade aproximada: 6.000 imagens

Número de classes: 97 (algumas categorias foram excluídas)

Tamanho das imagens: 224 × 224 pixels

Exclusões realizadas:

- BACKGROUND_Google

- Motorbikes

- airplanes

- Faces_easy

- Faces
---
### A divisão do dataset foi feita da seguinte forma:

- Treino: 70%

- Validação: 15%

- Teste: 15%
---
### Tecnologias Utilizadas

- Python

- Keras

- TensorFlow

- NumPy

- Matplotlib

- Redes Neurais Convolucionais (CNN)

- Transfer Learning

- Fine-tuning

- VGG16 (ImageNet)
---
### Estrutura do Projeto

Carregamento e pré-processamento das imagens

Normalização dos dados

Codificação one-hot dos rótulos

Treinamento de uma CNN do zero

Avaliação do modelo em conjunto de teste

Aplicação de Transfer Learning com VGG16

Congelamento das camadas convolucionais

Treinamento de uma nova camada de classificação

Comparação dos resultados entre os modelos

Metodologia
Modelo 1: CNN Treinada do Zero

Arquitetura baseada em múltiplas camadas convolucionais

Uso de ReLU, MaxPooling e Dropout

Camadas densas finais com Softmax

Função de perda: categorical_crossentropy

Otimizador: Adam

Modelo 2: Transfer Learning com VGG16

Utilização da VGG16 pré-treinada no ImageNet

Remoção da camada de classificação original

Inserção de uma nova camada Softmax com o número de classes do dataset

Congelamento das camadas pré-treinadas

Treinamento apenas da camada final

---
### Resultados

O modelo com Transfer Learning apresentou desempenho superior em comparação ao modelo treinado do zero.

Houve melhora significativa na acurácia e redução da perda de validação.

O uso de pesos pré-treinados demonstrou ser eficaz mesmo com um número limitado de imagens.

Gráficos de perda e acurácia ao longo das épocas foram gerados para análise comparativa.

Exemplo de Predição

O projeto inclui um exemplo de inferência utilizando uma imagem do dataset para obtenção das probabilidades de classificação com o modelo treinado.

---
### Objetivo Educacional

Este projeto foi desenvolvido com fins de estudo e prática, visando:

Consolidar conceitos de CNN

Compreender Transfer Learning e Fine-tuning

Trabalhar com classificação de imagens

Analisar impactos do uso de modelos pré-treinados
