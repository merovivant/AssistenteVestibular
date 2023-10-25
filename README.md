# Construção de um chatbot baseado em IA para responder dúvidas sobre o vestibular da Unicamp 2024.
Este projeto foi construído como parte de um processo seletivo de estágio para uma vaga que visa aplicar técnicas de Redes Neurais, Processamento de Linguagem Natural e Modelos de Linguagem de Grande Escala para construir chatbots apoiados por aprendizagem contextual.

## Introdução
Com o advento dos LLMs (Largue Language Models), as formas de interação com os computadores tiveram mudanças significativas e é possível notar o surgimento de inúmeros chatbots que interagem com o usuário de uma forma bem natural e até mesmo amigável. Esses novos modelos tem sido empregados em diversas áreas, substituindo por exemplo os antigos chatbots que utilizavam de respostas predefinidas para, buscando palavras-chave, responder os comandos dos usuários. Esses modelos mais antigos possuíam algumas limitações principalmente no que diz respeito a compreensão contextual e oferecer respostas a perguntas mais avançadas.

Apesar dos novos modelos baseados em apredizagem de máquina e processamento de linguagem natural corrigirem muitos dos problemas encontrados nos chatbots mais antigos, eles ainda existem algumas limitações. Esses modelos são treinados em um conjunto de dados específico, o qual apesar de massivamente grande ainda é limitado e finito. A resposta pra alguma pergunta específica pode não estar no conjunto de dados utilizado para treinar o modelo, como no exemplo a seguir:

![ChatGPT respondendo que não possui informação sobre a data do vestibular da Unicamp](Imagens/screenshot1.jpeg)

Para corrigir esse problema, esse projeto irá utilizar um paradigma chamado RAG (Retrieval Augmented Genetarion). Esse paradigma será o responsável por utilizando da fonte apropriada, recuperar de um documento específico informações relevantes para responder a questão do usuário. Com isso, essas informações relevantes são incluídas junto do prompt e o modelo de linguagem se torna mais apto para responder as perguntas no caso de uso específico em que está sendo empregado.

A seguir está a captura de tela conseguida após o desenvolvimento do projeto:
*Incluir captura de tela atualizada*

## Implementação
A implementação do projeto foi apoiada pelas seguintes bibliotecas e fonte de dados:

### Fonte de dados
[Resolução GR-031/2023, de 13/07/2023](https://www.pg.unicamp.br/norma/31594/0): Dispõe sobre o Vestibular Unicamp 2024 para vagas no ensino de Graduação

### Bibliotecas
[Pinecone](https://www.pinecone.io/): Base de dados gratuíta que possibilita armazenamento de vetores dos embeddings gerados pela OpenAI, apoiando a consulta contextual por meio da query e retornando os vetores mais relevantes.

[LangChain Python](https://python.langchain.com/docs/get_started/introduction): Biblioteca open-source voltada para desenvolvimento de aplicações apoiadas por modelos de linguagem natural. 
O LangChain foi utilizado para integração de todas as partes incluindo bibliotecas e APIs utilizadas no projeto, desde a recuperação, carregamento e transformação dos documentos de fonte de dados, integração com os modelos da OpenAI de Embeddings e geração de texto, recuperação contextual dos vetores armazenados no Pinecone e integração com o Streamlit na construção da interface gráfica.

[Streamlit](https://streamlit.io/): Biblioteca open-source voltada para construção facilitada de interfaces gráficas na forma de aplicações web apoiadas por machine learning e ciência de dados. Também disponibiliza o Streamlit Community Cloud para deploy e gerenciamento gratuíto dos aplicativos construídos com a biblioteca.
