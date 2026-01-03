""" Descrição do Desafio
Projeto de criação de uma base de dados e treinamento da rede YOLO . 

Seguindo os exemplos de aula, vamos rotular uma base de dados e aplicar o treinamento com a rede YOLO. 
Para essa tarefa será necessário utilizar o software LabelMe: http://labelme.csail.mit.edu/Release3.0/ para rotular as imagens. 
Também será necessário utilizar a rede YOLO, disponível em: https://pjreddie.com/darknet/yolo/. 
Para quem preferir e não quiser rotular uma base de dados, pode usar as imagens já rotuladas do COCO: https://cocodataset.org/#home. 
E para quem estiver utilizando um computador que não consiga rodar a rede YOLO, pode utilizar o transfer learning no COLAB: https://colab.research.google.com/drive/1lTGZsfMaGUpBG4inDIQwIJVW476ibXk_#scrollTo=j0t221djS1Gk. 
O trabalho deve conter pelo menos duas classes retreinadas para detecção, além das classes já treinadas previamente antes de realizar o transfer learning.  
Por meio da imagem é possível visualizar um exemplo de resultado esperado:  

STEP 0. Configure runtime to work with GPU
STEP 1. Connect your files to Google Drive
STEP 2. Connect the Colab notebook to Google Drive -> STEP 2. Check CUDA release version
STEP 3. Install cuDNN according to the current CUDA version
STEP 4. Installing Darknet -> STEP 4-A. 
Cloning and compiling Darkent. ONLY NEEDS TO BE RUN ON THE FIRST EXECUTION!!
In this step we'll clone the darkent repo and compile it.
    Clone Repo
    Compile Darknet
    Copy compiled version to Drive

STEP 4-B. Copying the compiled version of Darknet from Drive. UNCOMMENT AFTER FIRST EXECUTION

STEP 5. Runtime configuration finished!
Let's chek it out!

If you are running this notebook for the first time, you can run the following cells in order to check if everything goes as expected!
"""
#download files
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
# %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
  
  
def upload():
  #from google.colab import files
  uploaded = files.upload() 
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print ('saved file', name)
def download(path):
  from google.colab import files
  files.download(path)

