#pip install langchain pypdf openai chromadb tiktoken
#pypdf nos ayuda a cargar documentos pdf de manera local y luego convertirlos en documentos que puedan ser legibles por langchain
#openai es necesario por que vamos a utilizar un modelo de lenguaje de openai
#chromadb es una base de datos vectorial y la utilizaremos para crear indices y encontrar los documentos que mas sirven para reponder las preguntas del usuario
#tiktoken sirve para tokenizar y obtener informacion de los tokens del texto que ingresamos
import os  # Acceso al sistema operativo 
import requests  # Nos permite consumir la API y obtener los documentos
from decouple import config # Carga de variables de entorno
from langchain_community.document_loaders import PyPDFLoader  # Lectura de archivos PDF

OPENAI_API_KEY = config('OPENAI_API_KEY') # Trae la variable de entono que almacena la KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY # Guarda la llave en las variables del sistema operativo

urls = [
    'https://arxiv.org/pdf/2306.06031v1.pdf',
    'https://arxiv.org/pdf/2306.12156v1.pdf',
    'https://arxiv.org/pdf/2306.14289v1.pdf',
    'https://arxiv.org/pdf/2305.10973v1.pdf',
    'https://arxiv.org/pdf/2306.13643v1.pdf'
]

ml_papers = []

os.makedirs('papers', exist_ok=True) # Se encarga de crear la carpeta en caso de que no exista

for i, url in enumerate(urls): # Itera sobre cada indice (i) y URL en la lista de URLs
    response = requests.get(url) # Realiza una solicitud GET a la URL para obtener el contenido del archivo
    filename = os.path.join('papers', f'paper{i+1}.pdf') # Construye el nombre del archivo combinando la carpeta 'papers' y el nombre del archivo
    with open(filename, 'wb') as f: # Abre el archivo en modo de escritura binaria ('wb') y escribe el contenido de la respuesta (Si el archivo no existe se encarga de crearlo)
        f.write(response.content) # Escribe el contenido de la respuesta (el archivo descargado) en el archivo
        print(f'Descargado {filename}')

        loader = PyPDFLoader(filename) # Sirve para cargar la informacion del documento
        data = loader.load() # Convierte el documento PDF a la clase de langchain Document
        ml_papers.extend(data) # Guarda toda la informacion de los PDF en una gran lista donde cada elemento es una parte de los PDF

# Lo siguiente que se debe hacer es convertir el texto en numeros atraves de enbeddings
# Pero primero debemos tener en cuenta que los modelos embeddings solo aceptan cierto numero de tokens
# Entonces primero debemos partirlo en documentos mas pequeños antes de ingresarlo en modelos de embeddings
        
from langchain.text_splitter import RecursiveCharacterTextSplitter # Nos ayuda a partir textos muy largos en bloques mas cortos

text_splitter = RecursiveCharacterTextSplitter( # Configuramos el textsplitter
    chunk_size=1500, # Cada texto va a tener un tamaño de 1500 caracteres
    chunk_overlap=200, # Al inicio va a tener 200 caracteres que se repiten del final del fragmento anterior y al final va a tener 200 caracteres que se repetiran al inicio del siguiente fragmento( Esto lo hace para tener continuidad entre los fragmentos )
    length_function=len # Le indicamos que va a ser la longitud de un string
)

documents = text_splitter.split_documents(ml_papers) # Aplicamos el text splitter

# Ahora si podemos pasar el texto a un modelo de embeddings 

#pip install langchain_openai
from langchain_openai import OpenAIEmbeddings # Importamos el modelo de embeddings de OpenAI que se encargara de convertir las cadenas en vectores
from langchain_community.vectorstores import Chroma # Importamos la base de datos vectorial donde se almacenaran las cadenas convertidas en vectores

embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002') # Definimos el modelo de embedding que vamos a utilizar

vectorstore = Chroma.from_documents( # Hacemos una instancia de la base de datos
    documents=documents, # Indicamos los documentos que va a tener convertidos en numeros
    embedding=embeddings # Le indicamos el modelo de embedding que queremos que utilice
)

# Ahora convertimos la vectorstore en un retriever para que tambien pueda buscar informacion relevante
retriever = vectorstore.as_retriever(
    search_kwargs={"k" : 3} # Le indicamos que busque solamente los 3 documentos que mas se parecen al query
)

from langchain_openai import ChatOpenAI # Importamos el modelo de lenguaje que vamos a utilizar para resolver dudas y a partir de la informacion que le demos
from langchain.chains import RetrievalQA # Cadena de procesamiento para realizar preguntas y respuestas

chat = ChatOpenAI( # Inicialimos nuestro modelo
    openai_api_key=OPENAI_API_KEY, # API_KEY que obtuvimos de openai
    model_name='gpt-3.5-turbo', # Nombre del modelo que queremos utilizar
    temperature=0.0 # Le indicamos que conteste de manera muy precisa sin creatividad
)

qa_chain = RetrievalQA.from_chain_type( # Creamos la cadena
    llm=chat, # Le indicameos el modelo de chat que vamos a utilizar el cual creamos anteriormente
    chain_type='stuff', # Le indicamos el tipo de cadena, en esta caso es "stuff" que significa lo que quepa en el prompt
    retriever=retriever # Le indicamos el modulo de recuperacion que va a utilizar
)

query = "Que es fingpt?"
print(qa_chain.invoke(query))