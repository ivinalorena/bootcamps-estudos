"""projeto:
Descrição do Desafio
Seguindo o exemplo do algoritmo de binarização apresentado em nossa última aula, realize a implementação em Python para transformar 
uma imagem colorida para níveis de cinza (0 a 255) e para binarizada (0 e 255), preto e branco.  """

""" imagem_colorida = [
    [(255, 0, 0), (0, 255, 0)],
    [(0, 0, 255), (255, 255, 255)]
]
def rgb_para_cinza(imagem):
    imagem_cinza=[]

    for linha in imagem:
        nova_linha=[]
        for (r,g,b) in linha:
            cinza = int(0.299*r + 0.587*g + 0.114*b)
            nova_linha.append((cinza,cinza,cinza))
        imagem_cinza.append(nova_linha)

    return imagem_cinza

imagem_cinza = rgb_para_cinza(imagem_colorida)

print(imagem_cinza) """

""" from PIL import Image

imagem = Image.open("dio/gato.jfif")
imagem = imagem.convert("RGB")
largura, altura = imagem.size
imagem_cinza = Image.new("RGB", (largura, altura))

for x in range(largura):
    for y in range(altura):
        r, g, b = imagem.getpixel((x, y))

        cinza = int(0.299 * r + 0.587 * g + 0.114 * b)
        imagem_cinza.putpixel((x, y), (cinza, cinza, cinza))

imagem_cinza.save("imagem_cinza.jpeg")

print("Imagem convertida para escala de cinza com sucesso.")
 """

from PIL import Image


imagem = Image.open("dio/gato.jfif").convert("RGB")
largura, altura = imagem.size
imagem_binaria = Image.new("RGB", (largura, altura))

limiar = 128

for x in range(largura):
    for y in range(altura):
        r, g, b = imagem.getpixel((x, y))
        cinza = int(0.299 * r + 0.587 * g + 0.114 * b)

        if cinza >= limiar:
            imagem_binaria.putpixel((x, y), (255, 255, 255))  # branco
        else:
            imagem_binaria.putpixel((x, y), (0, 0, 0))        # preto


imagem_binaria.save("imagem_binaria.jpeg")

print("Imagem binarizada com sucesso.")
