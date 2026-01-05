""" 
Sistema de Recomendação por Imagens
====================================

Descrição do Desafio:
Sistema capaz de classificar imagens por sua similaridade e gerar recomendações
para o usuário de um site. O sistema indica produtos relacionados por sua 
aparência física (formato, cor, textura) e não por dados textuais.

Referência: https://colab.research.google.com/github/sparsh-ai/rec-tutorials/blob/master/_notebooks/2021-04-27-image-similarity-recommendations.ipynb
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
from typing import List, Tuple
import pickle


class ImageRecommendationSystem:
    """
    Sistema de recomendação baseado em similaridade visual de imagens.
    Utiliza uma rede ResNet50 pré-treinada para extrair features das imagens.
    """
    
    def __init__(self, model_name='resnet50'):
        """
        Inicializa o sistema de recomendação.
        
        Args:
            model_name: Nome do modelo pré-treinado a ser utilizado
        """
        print("Carregando modelo pré-treinado...")
        # Carrega ResNet50 sem a camada de classificação final
        self.model = ResNet50(
            weights='imagenet',
            include_top=False,
            pooling='avg'
        )
        print("Modelo carregado com sucesso!")
        
        self.image_features = {}
        self.image_paths = []
        
    def extract_features(self, img_path: str) -> np.ndarray:
        """
        Extrai features de uma imagem usando o modelo pré-treinado.
        
        Args:
            img_path: Caminho para a imagem
            
        Returns:
            Array numpy com as features extraídas
        """
        try:
            # Carrega e processa a imagem
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extrai features
            features = self.model.predict(img_array, verbose=0)
            
            # Normaliza o vetor de features
            features = features.flatten()
            features = features / np.linalg.norm(features)
            
            return features
        
        except Exception as e:
            print(f"Erro ao processar imagem {img_path}: {str(e)}")
            return None
    
    def build_index(self, image_directory: str, extensions: List[str] = None):
        """
        Constrói o índice de features para todas as imagens em um diretório.
        
        Args:
            image_directory: Diretório contendo as imagens
            extensions: Lista de extensões de arquivo aceitas
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        
        print(f"Indexando imagens do diretório: {image_directory}")
        
        image_dir = Path(image_directory)
        image_files = []
        
        # Busca todas as imagens no diretório
        for ext in extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))
        
        print(f"Encontradas {len(image_files)} imagens")
        
        # Extrai features de cada imagem
        for idx, img_path in enumerate(image_files):
            print(f"Processando {idx + 1}/{len(image_files)}: {img_path.name}")
            
            features = self.extract_features(str(img_path))
            
            if features is not None:
                self.image_features[str(img_path)] = features
                self.image_paths.append(str(img_path))
        
        print(f"Índice construído com {len(self.image_paths)} imagens")
    
    def find_similar_images(
        self, 
        query_image_path: str, 
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Encontra as imagens mais similares à imagem de consulta.
        
        Args:
            query_image_path: Caminho da imagem de consulta
            top_k: Número de recomendações a retornar
            
        Returns:
            Lista de tuplas (caminho_imagem, score_similaridade)
        """
        if len(self.image_paths) == 0:
            print("Erro: Índice vazio. Execute build_index() primeiro.")
            return []
        
        print(f"Buscando imagens similares a: {query_image_path}")
        
        # Extrai features da imagem de consulta
        query_features = self.extract_features(query_image_path)
        
        if query_features is None:
            print("Erro ao processar imagem de consulta")
            return []
        
        # Calcula similaridade com todas as imagens do índice
        similarities = {}
        
        for img_path in self.image_paths:
            if img_path == query_image_path:
                continue  # Ignora a própria imagem
            
            indexed_features = self.image_features[img_path]
            
            # Calcula similaridade de cosseno
            similarity = cosine_similarity(
                query_features.reshape(1, -1),
                indexed_features.reshape(1, -1)
            )[0][0]
            
            similarities[img_path] = similarity
        
        # Ordena por similaridade (maior para menor)
        sorted_similarities = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Retorna top_k resultados
        return sorted_similarities[:top_k]
    
    def save_index(self, filepath: str):
        """
        Salva o índice de features em arquivo.
        
        Args:
            filepath: Caminho do arquivo para salvar
        """
        index_data = {
            'image_features': self.image_features,
            'image_paths': self.image_paths
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"Índice salvo em: {filepath}")
    
    def load_index(self, filepath: str):
        """
        Carrega o índice de features de um arquivo.
        
        Args:
            filepath: Caminho do arquivo para carregar
        """
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.image_features = index_data['image_features']
        self.image_paths = index_data['image_paths']
        
        print(f"Índice carregado com {len(self.image_paths)} imagens")


def exemplo_uso():
    """
    Exemplo de uso do sistema de recomendação por imagens.
    """
    # Inicializa o sistema
    sistema = ImageRecommendationSystem()
    
    # Constrói o índice com imagens de um diretório
    # ALTERE O CAMINHO PARA SEU DIRETÓRIO DE IMAGENS
    diretorio_imagens = "./imagens_produtos"
    
    if os.path.exists(diretorio_imagens):
        sistema.build_index(diretorio_imagens)
        
        # Opcional: Salva o índice para uso futuro
        sistema.save_index("index_imagens.pkl")
        
        # Busca imagens similares
        # ALTERE PARA O CAMINHO DA SUA IMAGEM DE CONSULTA
        imagem_consulta = "./imagens_produtos/produto1.jpg"
        
        if os.path.exists(imagem_consulta):
            resultados = sistema.find_similar_images(
                imagem_consulta, 
                top_k=5
            )
            
            print("\n" + "="*60)
            print("RECOMENDAÇÕES DE PRODUTOS SIMILARES")
            print("="*60)
            
            for idx, (img_path, score) in enumerate(resultados, 1):
                print(f"{idx}. {Path(img_path).name}")
                print(f"   Similaridade: {score:.4f}")
                print(f"   Caminho: {img_path}")
                print()
        else:
            print(f"Imagem de consulta não encontrada: {imagem_consulta}")
    else:
        print(f"Diretório não encontrado: {diretorio_imagens}")
        print("\nPara usar este sistema:")
        print("1. Crie um diretório com suas imagens de produtos")
        print("2. Atualize a variável 'diretorio_imagens' com o caminho correto")
        print("3. Execute o código novamente")


if __name__ == "__main__":
    # Exemplo de uso básico
    print("="*60)
    print("SISTEMA DE RECOMENDAÇÃO POR IMAGENS")
    print("="*60)
    print()
    
    exemplo_uso()
    
    # Exemplo de como carregar um índice já criado
    """
    sistema = ImageRecommendationSystem()
    sistema.load_index("index_imagens.pkl")
    resultados = sistema.find_similar_images("nova_imagem.jpg", top_k=5)
    """

