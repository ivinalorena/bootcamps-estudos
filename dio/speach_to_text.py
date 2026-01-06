""" Criando um sistema de assistência virtual do zero. 
Neste projeto deve ser desenvolvido um sistema de assistência virtual, utilizando PLN (Processamento de Linguagem Natural), com base nas bibliotecas apresentadas durante o curso. O sistema deve obedecer aos seguintes requisitos: 
Um módulo para transformação de texto em áudio (text to speech); 
Um módulo para transformação de fala (linguagem natural humana) em texto (speech to text); 

O módulo 2, deve acionar por comando de voz algumas funções automatizadas, como por exemplo: abrir uma pesquisa no Wikipedia, abrir o Youtube, apresentar a localização da farmácia mais próxima. 
Para realizar este projeto, podem ser utilizada todas as bibliotecas apresentadas no curso, principalmente a biblioteca Speech recognition em Python.  
Para auxiliar no projeto, estão disponíveis dois exemplos, um para text to speech e outro para speech to text. Ambos podem ser encontrados no Github, seguindo os links abaixo:  """

# Text to Speach com Processamento de Linguagem Natural: 
from gtts import gTTS #pip install gTTS
text_to_say = "how are you doing?"
language = "en"

gtts_object = gTTS(text=text_to_say, lang=language, slow=False)
#gtts_object.save("test.mp3")
gtts_object.save('audio_teste.wav')

""" from IPython.display import Audio
Audio('/content/gtts.wav')
 """

#Fazer o inverso - ou seja, fazer com que o computador "escute" e interprete o que o usuário está falando
""" !pip install SpeechRecognition
!pip install playsound
!pip install pyjokes
!pip install wikipedia
!pip install pyaudio """

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


# pegar o mic 
def get_audio():
  r = sr.Recognizer()
  with sr.Microphone() as source:
    r.pause_threshold = 1
    # espere um segundo 
    r.adjust_for_ambient_noise(source, durantion=1)
    audio = r.listen(source)
    said = ""

    try:
      said = r.reconize_google(audio) #, language="en-US")
      print(said)

    except Exception as e:
      #print("Exception: " + str(e))
      speak("Desculpe, eu não entendi isso.")
  return said

def speak(text):
  tss=gTTS(text=text, lang="en")
  filename="voz.mp3"
  try:
    os.remove(filename)
  except OSError:
    pass

  tss.save(filename)
  playsound.playsound(filename)

# text = get_audio()
# speak(text)

while True:
    print("Estou escutando")
    text = get_audio().lower()
    if 'youtube' in text:
        speak("Abrindo o Youtube")
        url = "https://www.youtube.com"
        webbrowser.get().open(url)

    elif 'pesquisar' in text:
        speak("O que você gostaria de pesquisar?")
        query = text.replace("pesquisar", "")
        result = wikipedia.summary(query, sentences=2)
        speak("de acordo com a wikipedia") #speak(result)
        print(result)
        speak(result)

    elif 'piada' in text:
        speak(pyjokes.get_joke())

    elif 'farmacia' in text:
       speak("Buscando farmácias próximas a sua região...")
    elif 'sair' in text:
        speak("Saindo do programa")
        exit()
    else:
        speak("Desculpe, não entendi")