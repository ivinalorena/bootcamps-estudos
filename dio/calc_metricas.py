"""Descrição do Desafio
Cálculo de Métricas de Avaliação de Aprendizado 

Neste projeto, vamos calcular as principais métricas para avaliação de modelos de classificação de dados, como acurácia, sensibilidade (recall), 
especificidade, precisão e
F-score. Para que seja possível implementar estas funções, você deve utilizar os métodos e suas fórmulas correspondentes (Tabela 1). 

Para a leitura dos valores de VP, VN, FP e FN, será necessário escolher uma matriz de confusão para a base dos cálculos. Essa matriz você pode escolher 
de forma arbitraria, pois nosso objetivo é entender como funciona cada métrica. 

sensibilidade: VP / (VP + FN)
especificidade: VN / (VN + FP)
precisão: VP / (VP + FP)
acurácia: (VP + VN) / (VP + VN + FP + FN)
F-score: 2 * (precisão * sensibilidade) / (precisão + sensibilidade)
"""

def calcular_metricas(vp, vn, fp, fn):
    sensibilidade = vp / (vp + fn) if (vp + fn) != 0 else 0
    especificidade = vn / (vn + fp) if (vn + fp) != 0 else 0
    precisao = vp / (vp + fp) if (vp + fp) != 0 else 0
    acurácia = (vp + vn) / (vp + vn + fp + fn) if (vp + vn + fp + fn) != 0 else 0
    f_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) != 0 else 0

    return {
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "precisão": precisao,
        "acurácia": acurácia,
        "F-score": f_score
    }

