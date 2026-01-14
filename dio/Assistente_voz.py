#Assistente de Voz Multi-Idiomas Com Whisper e ChatGPT

# Referência: https://gist.github.com/korakot/c21c3476c024ad6d56d5f48b0bca92be

from IPython.display import Audio, display, Javascript
from google.colab import output
from base64 import b64decode

# Código JavaScript para gravar áudio do usuário usando a "MediaStream Recording API"
RECORD = """
const sleep  = time => new Promise(resolve => setTimeout(resolve, time))
const b2text = blob => new Promise(resolve => {
  const reader = new FileReader()
  reader.onloadend = e => resolve(e.srcElement.result)
  reader.readAsDataURL(blob)
})
var record = time => new Promise(async resolve => {
  stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  recorder = new MediaRecorder(stream)
  chunks = []
  recorder.ondataavailable = e => chunks.push(e.data)
  recorder.start()
  await sleep(time)
  recorder.onstop = async ()=>{
    blob = new Blob(chunks)
    text = await b2text(blob)
    resolve(text)
  }
  recorder.stop()
})
"""

def record(sec=5):
  # Executa o código JavaScript para gravar o áudio
  display(Javascript(RECORD))
  # Recebe o áudio gravado como resultado do JavaScript
  js_result = output.eval_js('record(%s)' % (sec * 1000))
   # Decodifica o áudio em base64
  audio = b64decode(js_result.split(',')[1])
  # Salva o áudio em um arquivo
  file_name = 'request_audio.wav'
  with open(file_name, 'wb') as f:
    f.write(audio)
  # Retorna o caminho do arquivo de áudio (pasta padrão do Google Colab)
  return f'/content/{file_name}'

# Grava o áudio do usuário por um tempo determinado (padrão 5 segundos)
print('Ouvindo...\n')
record_file = record()

# Exibe o áudio gravado
display(Audio(record_file, autoplay=False))

import whisper

# Selecione o modelo do Whisper que melhor atenda às suas necessidades:
# https://github.com/openai/whisper#available-models-and-languages
model = whisper.load_model("small")

# Transcreve o audio gravado anteriormente.
result = model.transcribe(record_file, fp16=False, language=language)
transcription = result["text"]
print(transcription)

import os

# Documentação Oficial da API OpenAI: https://platform.openai.com/docs/api-reference/introduction
# Informações sobre o Período Gratuito: https://help.openai.com/en/articles/4936830

# Para gerar uma API Key:
# 1. Crie uma conta na OpenAI
# 2. Acesse a seção "API Keys"
# 3. Clique em "Create API Key"
# Link direto: https://platform.openai.com/account/api-keys

os.environ['OPENAI_API_KEY'] = ''

import openai

# Configura a chave de API da OpenAI usando a variável de ambiente 'OPENAI_API_KEY'
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Envia uma requisição à API do ChatCompletion usando o modelo GPT-3.5 Turbo
# Lembrando que, a variável 'transcription' contém a transcrição do nosso áudio.
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[ { "role": "user", "content": transcription } ]
)

# Obtém a resposta gerada pelo ChatGPT
chatgpt_response = response.choices[0].message.content
print(chatgpt_response)

#!pip install gTTS

from gtts import  gTTS

# Cria um objeto gTTS com a resposta gerada pelo ChatGPT e a língua que será sintetizada em voz (variável "language").
gtts_object = gTTS(text=chatgpt_response, lang=language, slow=False)

# Salva o áudio da resposta no arquivo especificado (pasta padrão do Google Colab)
response_audio = "/content/response_audio.wav"
gtts_object.save(response_audio)

# Reproduz o áudio da resposta salvo no arquivo
display(Audio(response_audio, autoplay=True))