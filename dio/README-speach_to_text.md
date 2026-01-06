## Text to Speach com Processamento de Linguagem Natural:
### Projeto desenvolvido no bootcamp da DIO + BairesDev - Machine Learning Practitioner 

> Objetivo computador "falar" com o texto digitado pelo usuário (mudar a variável de text = "" e language ="") 
- Exemplo:
```Python
from gtts import gTTS #pip install gTTS
text_to_say = "how are you doing?" #
language = "en" #
gtts_object = gTTS(text=text_to_say, lang=language, slow=False)
gtts_object.save('audio_teste.wav')
```
- Para fazer o inverso:
ou seja, fazer com que o computador "escute" e interprete o que o usuário está falando

```Python
# importações necessárias para o projeto
import speech_recognition as sr
from gtts import gTTS
import os
import datetime
import playsound
import pyjokes
import wikipedia
import pyaudio
import webbrowser

# 1. iniciar o mic
# 2. definir a função speak utilizando a biblioteca gTTS
# 3. estrutura de repetição com as condicionais
```
---
#### Link do colab para o projeto:
https://colab.research.google.com/drive/1AGle5Wh5T6aABiNcrYg2X6zSnEFqhkCZ?usp=sharing

## Stack utilizada
**Python**

## Autores

- [@ivinalorena](https://github.com/ivinalorena)