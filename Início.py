import  streamlit as st
import streamlit as st 
import pandas as pd 
# Import libraries
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import matplotlib.pyplot as plt

from gensim.corpora import Dictionary
from gensim.models import LdaModel

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
    
# Importing libraries
import pandas as pd
import numpy as np
import nltk
import spacy
import sklearn

import re
import contractions
nltk.download('punkt')
nltk.download('all')
from nltk.corpus import stopwords
from nltk import word_tokenize
# Used in Lemmatization
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from nltk import pos_tag

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt
import seaborn as sns
# Wordcloud to check the most used words and add appropriate ones to the stopwords list
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import warnings
warnings.filterwarnings("ignore")

from wordcloud import WordCloud







# Tempo
with st.container():
    
    import time

    progress_text = "Carregando dados. Por favor aguarde."
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    time.sleep(2)
    my_bar.empty()








# Cabeçalho
with st.container():
    st.subheader("Trabalho de Introdução à Ciência de Dados")
    st.write("Mestrado em Ciência de Dados para Ciências Sociais da Universidade de Aveiro")
    st.title("Análise de E-Commerce")
    st.write("---")
    
    
    
    
    
    
       
    
# Imagem do e-commerce e texto de contextualização e problema
with st.container():
    import streamlit as st
    from PIL import Image
    
    #Abrir a imagem com a PIL
    image = Image.open('ecommerce_image.png')

    # Adicionar a imagem usando st.image()
    st.image(image, width=690, caption='')
    
    st.write("---")
    
    st.subheader("Contextualização do trabalho")
    
    tab1, tab2, tab3 = st.tabs(["Geral", "Específica", "Motivação"])
    
    with tab1:
    
        st.write("Trabalho com o intuito de perspetivar o impacto, consumo e distribuição de compradores digitais. Tem como fundamento artigos escritos da base de dados Scopus, aos quais foram aplicadas técnicas de Natural Processed Language (NPL), pré-processamento de dados, modelagem.")
    
    with tab2:
        
        st.write("Trabalho com o intuito de perspetivar o impacto, consumo e distribuição de compradores digitais. Tem como fundamento artigos escritos da base de dados Scopus, aos quais foram aplicadas técnicas de Natural Processed Language (NPL), pré-processamento de dados, modelagem, Clusterização e cálculos estatísticos, recorrendo a diversos tipos de gráficos, tabelas e wordclouds de forma a retirar os melhores insights. Trabalho com o intuito de perspetivar o impacto, consumo e distribuição de compradores digitais. Tem como fundamento artigos escritos da base de dados Scopus, aos quais foram aplicadas técnicas de Natural Processed Language (NPL), pré-processamento de dados, modelagem, Clusterização e cálculos estatísticos, recorrendo a diversos tipos de gráficos, tabelas e wordclouds de forma a retirar os melhores insights.")
    
    with tab3:
        
        st.write("Compreender o intuito de perspetivar o impacto, consumo e distribuição de compradores digitais. Tem como fundamento artigos escritos da base de dados Scopus, aos quais foram aplicadas técnicas de Natural Processed Language (NPL), pré-processamento de dados, modelagem, Clusterização e cálculos estatísticos, recorrendo a diversos tipos de gráficos, tabelas e wordclouds de forma a retirar os melhores insights.")
        
    st.write("---")
    
    
    
    





  
    
    
# Importação e processamento do abstract
with st.container():
    
    # IMPORTAR ARQUIVO
    df = pd.read_csv('scopus.csv')
    

    # Get the number of duplicates
    duplicate = df['Abstract'].duplicated().sum()
    
    # Remove duplicate rows
    df = df.drop_duplicates(subset=['Abstract'])
    
    def clean_text(text_string, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''):

        # Cleaning the urls
        string = re.sub(r'https?://\S+|www\.\S+', '', text_string)

        # Cleaning the html elements
        string = re.sub(r'<.*?>', '', string)

        # Removing the punctuations
        string = re.sub(r'[^\w\s]', '', string)

        # Converting the text to lower
        string = string.lower()

        # Removing stop words
        filtered_words = [word for word in string.split() if word not in stopwords.words('english')]

        # Custom stop words list
        customlist = ['not', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                    "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                    "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                    "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

        # Tokenization
        tokens = word_tokenize(' '.join(filtered_words))

        # Remove numbers
        tokens = [word for word in tokens if word.isalpha()]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

        # Stemming
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

        # Fix contractions
        final_string = ' '.join([contractions.fix(word) for word in stemmed_words])

        return final_string
    
    #converts all the values in a specific columns of the DataFrame data to strings 
    df["Abstract"] = df["Abstract"].astype(str) 

    #Applying a Text Cleaning Function
    df['clean_Abstract'] = df['Abstract'].apply(clean_text)
    
    df = df[df['Abstract'] != '[No abstract available]']
    
    def remove_specific_words(text):
        words_to_remove = ['use', 'research', 'studi', 'provid', 'author', 'using', 'model', 'analysi', 'paper', 'analyz', 'cluster', 'data']
        return ' '.join([word for word in text.split() if word.lower() not in words_to_remove])

    # Aplicar a função na coluna 'clean_Abstract'
    df['clean_Abstract'] = df['clean_Abstract'].apply(remove_specific_words)
    









# nuvem de palavras, frequencia e falta fazer uma treemap
with st.container():
    st.subheader("Palavras mais relevantes para o contexto do trabalho")
    tab1, tab2 = st.tabs(["Wordcloud", "Frequência"])

    # Gerar a wordcloud com o abstract
    with tab1:
        
        # Criar a nuvem de palavras
        wordcloud = WordCloud(
            background_color='white', max_words=10000, width=800, height=600, stopwords=STOPWORDS
        ).generate(" ".join(df["clean_Abstract"]))

        # Criar o gráfico de nuvem de palavras usando Matplotlib e exibi-lo no Streamlit
        fig, ax = plt.subplots(figsize=(16, 13))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title("", fontsize=20)
        ax.axis('off')

        # Exibir o gráfico no Streamlit
        st.pyplot(fig)
        
    




    # gerar uma st.progress para as 10 palavras mais frequentes
    with tab2:

        from collections import Counter
        import time
       
        
        # Obter as palavras mais frequentes
        word_freq = Counter(" ".join(df["clean_Abstract"]).split())
        top_words = word_freq.most_common(10)

        # Encontrar a frequência máxima para normalizar o preenchimento da barra de progresso
        max_freq = top_words[0][1]

        # Layout com texto acima e barra de progresso abaixo para cada palavra
        for word, freq in top_words:
            st.write(f"**{word}**: {freq} frequência")  # Exibir palavra e frequência acima da barra de progresso
            progress_bar = st.progress(0)  # Criar a barra de progresso
            
            for progress_count in range(freq + 1):
                progress_percentage = progress_count / max_freq  # Calcular a porcentagem de progresso
                progress_bar.progress(progress_percentage)  # Atualizar a barra de acordo com a porcentagem de progresso
                time.sleep(0.00)  # Pequeno intervalo para controlar a velocidade da barra

















# algo interativo com os insigths

st.write("---")
with st.container():
    
    st.caption("Problema em questão")
    st.subheader("Como é representado o E-Commerce ao longo do tempo? Quais as alterações e enfoques de abordagem?")
    
    
    import plotly.express as px


    # Selecionar o parametro para ver e caixas 
          
    modificador = st.selectbox("Selecione o indicador que deseja visualizar", ["Tipologia dos Documentos por Ano", "Frequência de Artigos por Ano", "Palavras com Maior Nível de Importância (N-Grams) por Ano", "Palavras com Maior Nível de Importância (Bigrams) por Ano", "Análise de Sentimento do Conteúdo por Ano", "Classificação Gramatical do Conteúdo por Ano", "Influênia da Covid na Produção do Conhecimento por Ano"])
    
    col1, col2, col3 = st.columns(3)
    
    with col1:   
        tabela = st.checkbox('Adicionar tabela')
        
    
    
    with col2:   
        grafico = st.checkbox('Adicionar gráfico')
     
    
        
        
        
    if modificador == "Tipologia dos Documentos por Ano":
        
        # Obter contagem dos tipos de documento para cada ano
        contagem_tipos_documento_por_ano = df.groupby(['Year', 'Document Type']).size().reset_index(name='Count')

        #Criar o treemap com Plotly Express
        fig = px.treemap(contagem_tipos_documento_por_ano, path=['Year', 'Document Type'], values='Count',
                        hover_data=['Count'], branchvalues='total', width=700, height=800)
        fig.update_traces(root_color="white", selector=dict(type='treemap'))
        st.plotly_chart(fig)
        
        
        
            
        # dados da tabela caso seja selecionada, so para a tipologia
        if tabela:

            # Criar uma tabela para mostrar a contagem de tipos de documento por ano
            st.write('Frequência de Tipos de Documento por Ano')
            for year in df['Year'].unique():
                st.write(f"Ano: {year}")
                df_year = contagem_tipos_documento_por_ano[contagem_tipos_documento_por_ano['Year'] == year]
                st.write(df_year)

            # Criar uma tabela com a contagem total de cada tipo de documento
            st.write('Total de Tipos de Documento')
            df_total = contagem_tipos_documento_por_ano.groupby('Document Type')['Count'].sum().reset_index(name='Total')
            st.write(df_total)


        
       
        # dados em grafico de bolhas
        if grafico:

            # Criar o gráfico de bolhas
            fig = px.scatter(contagem_tipos_documento_por_ano, x='Year', y='Document Type', size='Count', color='Document Type',
                            hover_name='Document Type', hover_data={'Year': True, 'Document Type': False, 'Count': True},
                            width=900, height=600)
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig)

            










    if modificador == "Frequência de Artigos por Ano":
        df = pd.read_csv('scopus.csv')
        
        # Obter contagem total de documentos para cada ano
        contagem_documentos_por_ano = df['Year'].value_counts().reset_index()
        contagem_documentos_por_ano.columns = ['Year', 'Count']

        # Criar o treemap com Plotly Express
        fig = px.treemap(contagem_documentos_por_ano, path=['Year'], values='Count',
                        hover_data=['Count'], branchvalues='total', width=700, height=800)
        fig.update_traces(root_color="white", selector=dict(type='treemap'))
        st.plotly_chart(fig)
        
        
        
        if tabela:

            # Ordenar os dados pelo ano
            contagem_documentos_por_ano = contagem_documentos_por_ano.sort_values(by='Ano')

            # Exibir a tabela
            st.write("Contagem de Documentos por Ano:")
            st.write(contagem_documentos_por_ano)
            
            
            
        if grafico:

            # Criar o gráfico de bolhas
            fig = px.scatter(contagem_documentos_por_ano, x='Year', y='Count', size='Count', color='Count',
                            hover_name='Year', hover_data={'Year': True, 'Count': True},
                            width=900, height=600)
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
            st.plotly_chart(fig)
    
    
    
    
    
    
    
    
    
    
    
    if modificador == "Palavras com Maior Nível de Importância (N-Grams) por Ano":
        
        df = pd.read_csv('scopus.csv')
         
         # clean dataset
         
         # Get the number of duplicates
        duplicate = df['Abstract'].duplicated().sum()

        # Remove duplicate rows
        df = df.drop_duplicates(subset=['Abstract'])
        
        def clean_text(text_string, punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''):

            # Cleaning the urls
            string = re.sub(r'https?://\S+|www\.\S+', '', text_string)

            # Cleaning the html elements
            string = re.sub(r'<.*?>', '', string)

            # Removing the punctuations
            string = re.sub(r'[^\w\s]', '', string)

            # Converting the text to lower
            string = string.lower()

            # Removing stop words
            filtered_words = [word for word in string.split() if word not in stopwords.words('english')]

            # Custom stop words list
            customlist = ['not', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn',
                        "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                        "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn',
                        "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

            # Tokenization
            tokens = word_tokenize(' '.join(filtered_words))

            # Remove numbers
            tokens = [word for word in tokens if word.isalpha()]

            # Lemmatization
            lemmatizer = WordNetLemmatizer()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

            # Stemming
            stemmer = PorterStemmer()
            stemmed_words = [stemmer.stem(word) for word in lemmatized_words]

            # Fix contractions
            final_string = ' '.join([contractions.fix(word) for word in stemmed_words])

            return final_string
        
        #converts all the values in a specific columns of the DataFrame data to strings 
        df["Abstract"] = df["Abstract"].astype(str) 

        #Applying a Text Cleaning Function
        df['clean_Abstract'] = df['Abstract'].apply(clean_text)
        
        df = df[df['Abstract'] != '[No abstract available]']
        
        #... aims to create a new DataFrame called selected_columns that contains only the "Abstract" and "clean_Abstract" columns from the original DataFrame df. Then, it displays the first 20 rows of this new DataFrame using the .head(20) method.
        selected_columns = df[['Abstract', 'clean_Abstract']]
        
        def remove_specific_words(text):
            words_to_remove = ['use', 'research', 'studi', 'provid', 'author', 'using', 'model', 'analysi', 'paper', 'analyz', 'cluster', 'data']
            return ' '.join([word for word in text.split() if word.lower() not in words_to_remove])

        # Aplicar a função na coluna 'clean_Abstract'
        df['clean_Abstract'] = df['clean_Abstract'].apply(remove_specific_words)
    

        import streamlit as st
        import pandas as pd
        import plotly.express as px
        from nltk.tokenize import word_tokenize
        from collections import Counter
        
        # Lista para armazenar os dados do treemap
        data = []

        # Obtendo os anos únicos do DataFrame
        anos_unicos = df['Year'].unique()

        # Loop pelos anos
        for ano in anos_unicos:
            # Filtrando os dados para o ano específico
            dados_ano = df[df['Year'] == ano]
            
            # Concatenando todos os abstracts do ano em uma única string
            texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
            
            # Tokenização das palavras
            palavras_tokenizadas = word_tokenize(texto_ano)
            
            # Contagem das palavras e seleção das 10 mais comuns
            contagem_palavras = Counter(palavras_tokenizadas)
            palavras_comuns = contagem_palavras.most_common(10)
            
            # Adicionando os dados para o treemap
            for palavra, contagem in palavras_comuns:
                data.append({'Year': ano, 'Palavra': palavra, 'Contagem': contagem})

        # Criando o DataFrame a partir dos dados coletados
        df_treemap = pd.DataFrame(data)

        # Criando o treemap com Plotly Express
        fig = px.treemap(df_treemap, path=['Year', 'Palavra'], values='Contagem',
                       width=700, height=800)

        # Exibindo o treemap no Streamlit
        st.plotly_chart(fig)

        
        
        if tabela:
                
            # Lista para armazenar os dados das palavras mais frequentes globalmente
            palavras_mais_frequentes_global = []

            # Obtendo os anos únicos do DataFrame
            anos_unicos = df['Year'].unique()

            # Dicionário para armazenar as palavras mais frequentes por ano
            palavras_por_ano = {}

            # Loop pelos anos
            for ano in anos_unicos:
                # Filtrando os dados para o ano específico
                dados_ano = df[df['Year'] == ano]
                
                # Concatenando todos os abstracts do ano em uma única string
                texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
                
                # Tokenização das palavras
                palavras_tokenizadas = word_tokenize(texto_ano)
                
                # Contagem das palavras e seleção das 10 mais comuns por ano
                contagem_palavras = Counter(palavras_tokenizadas)
                palavras_comuns = contagem_palavras.most_common(10)
                
                # Armazenando as palavras mais frequentes por ano no dicionário
                palavras_por_ano[ano] = [palavra for palavra, _ in palavras_comuns]
                
                # Armazenando as palavras mais frequentes globalmente
                palavras_mais_frequentes_global.extend(palavra for palavra, _ in palavras_comuns)

            # Contagem das palavras mais frequentes globalmente
            contagem_palavras_global = Counter(palavras_mais_frequentes_global)
            palavras_comuns_global = contagem_palavras_global.most_common(10)

            # Criando as tabelas
            tabelas_ano = {}
            for ano, palavras in palavras_por_ano.items():
                df_ano = pd.DataFrame({'Palavras Mais Frequentes': palavras})
                tabelas_ano[ano] = df_ano

            # Tabela com as 10 palavras mais frequentes globalmente
            df_palavras_globais = pd.DataFrame({'Palavras Mais Frequentes (Global)': [palavra for palavra, _ in palavras_comuns_global]})

            # Exibindo as tabelas no Streamlit
            for ano, tabela in tabelas_ano.items():
                st.write(f"Tabela para o ano {ano}")
                st.write(tabela)

            st.write("As 10 palavras mais frequentes")
            st.write(df_palavras_globais)
    
    
    
        if grafico:
            
            # Lista para armazenar os dados das palavras mais frequentes globalmente
            palavras_mais_frequentes_global = []

            # Lista para armazenar os dados das palavras por ano
            dados_bolha = []

            # Obtendo os anos únicos do DataFrame
            anos_unicos = df['Year'].unique()

            # Loop pelos anos
            for ano in anos_unicos:
                # Filtrando os dados para o ano específico
                dados_ano = df[df['Year'] == ano]
                
                # Concatenando todos os abstracts do ano em uma única string
                texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
                
                # Tokenização das palavras
                palavras_tokenizadas = word_tokenize(texto_ano)
                
                # Contagem das palavras e seleção das 10 mais comuns por ano
                contagem_palavras = Counter(palavras_tokenizadas)
                palavras_comuns = contagem_palavras.most_common(10)
                
                # Adicionando os dados para o gráfico de bolhas
                for palavra, contagem in palavras_comuns:
                    dados_bolha.append({'Year': ano, 'Palavra': palavra, 'Contagem': contagem})
                    
                # Armazenando as palavras mais frequentes globalmente
                palavras_mais_frequentes_global.extend(palavra for palavra, _ in palavras_comuns)

            # Criando um DataFrame com os dados das palavras por ano
            df_bolha = pd.DataFrame(dados_bolha)

            # Criando o gráfico de bolhas com Plotly Express
            fig = px.scatter(df_bolha, x='Year', y='Palavra', size='Contagem', color='Contagem',
                            hover_name='Palavra', hover_data={'Year': True, 'Palavra': False, 'Contagem': True},
                            width=900, height=600)
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

            # Exibindo o gráfico no Streamlit
            st.plotly_chart(fig)
    
    
    
    
    
    
    
    
    
    
    
    
    
    if modificador == "Palavras com Maior Nível de Importância (Bigrams) por Ano":
        
        from nltk import bigrams
        
        # Lista para armazenar os dados do treemap
        data = []

        # Obtendo os anos únicos do DataFrame
        anos_unicos = df['Year'].unique()

        # Loop pelos anos
        for ano in anos_unicos:
            # Filtrando os dados para o ano específico
            dados_ano = df[df['Year'] == ano]
            
            # Concatenando todos os abstracts do ano em uma única string
            texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
            
            # Tokenização das palavras
            palavras_tokenizadas = word_tokenize(texto_ano)

            # Verificar se há pelo menos dois elementos para criar bigrams
            if len(palavras_tokenizadas) >= 2:
                # Criando bigrams
                lista_bigrams = list(bigrams(palavras_tokenizadas))
            else:
                lista_bigrams = []
            
            # Criando bigrams
            lista_bigrams = list(bigrams(palavras_tokenizadas))
            
            # Contagem dos bigrams e seleção dos 10 mais comuns
            contagem_bigrams = Counter(lista_bigrams)
            bigrams_comuns = contagem_bigrams.most_common(10)
            
            # Adicionando os dados para o treemap
            for bigram, contagem in bigrams_comuns:
                data.append({'Year': ano, 'Bigram': bigram, 'Contagem': contagem})

        # Criando o DataFrame a partir dos dados coletados
        df_treemap_bigrams = pd.DataFrame(data)

        # Criando o treemap com Plotly Express
        fig = px.treemap(df_treemap_bigrams, path=['Year', 'Bigram'], values='Contagem',
                    width=700, height=800)

        # Exibindo o treemap no Streamlit
        st.plotly_chart(fig)
        
    
    
    
    
        if tabela: 
            
            # Lista para armazenar os dados dos bigramas
            data_bigrams = []

            # Obtendo os anos únicos do DataFrame
            anos_unicos = df['Year'].unique()

            # Loop pelos anos
            for ano in anos_unicos:
                # Filtrando os dados para o ano específico
                dados_ano = df[df['Year'] == ano]
                
                # Concatenando todos os abstracts do ano em uma única string
                texto_ano = ' '.join(dados_ano['clean_Abstract'].values)
                
                # Tokenização das palavras
                palavras_tokenizadas = word_tokenize(texto_ano)
                
                # Obter bigramas
                lista_bigrams = list(bigrams(palavras_tokenizadas))
                
                # Contagem de bigramas
                contagem_bigrams = Counter(lista_bigrams)
                
                # 10 bigramas mais comuns
                mais_comuns_bigrams = contagem_bigrams.most_common(10)
                
                # Adicionando os dados dos bigramas ao DataFrame
                for bigrama, contagem in mais_comuns_bigrams:
                    data_bigrams.append({'Year': ano, 'Bigrama': ' '.join(bigrama), 'Contagem': contagem})

            # Criando o DataFrame com os dados coletados
            df_bigrams = pd.DataFrame(data_bigrams)

            # Criando tabelas para cada ano
            for ano in anos_unicos:
                st.write(f"Bigrams para o ano {ano}")
                tabela_ano = df_bigrams[df_bigrams['Year'] == ano]
                st.write(tabela_ano)

            # Tabela geral de bigramas
            st.write("Bigrams mais relevantes")
            tabela_geral = df_bigrams
            st.write(tabela_geral)
            
            
            
            
        
        if grafico:
            
            # Criando o gráfico de bolhas com Plotly Express
        
            fig = px.scatter(df_bigrams, x='Year', y='Bigrama', size='Contagem', color='Contagem',
                            hover_name='Bigrama', hover_data={'Year': True, 'Bigrama': False, 'Contagem': True},
                            width=900, height=600)

            # Atualizando a aparência do gráfico de bolhas
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

            # Exibindo o gráfico de bolhas no Streamlit
            st.plotly_chart(fig)
        
        
            
            
            
            
            
            
            
            
            
    
    if modificador == "Análise de Sentimento do Conteúdo por Ano":
  
        from textblob import TextBlob

        # Função para análise de sentimento
        def analyze_sentiment(text):
            
            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0:
                return 'Positivo'
            elif analysis.sentiment.polarity == 0:
                return 'Neutro'
            else:
                return 'Negativo'

        # Aplicar a função de análise de sentimento ao campo "clean_Abstract"
        df['Sentiment'] = df['clean_Abstract'].apply(analyze_sentiment)

        # Contagem dos resultados da análise de sentimento por ano
        sentiment_counts = df.groupby(['Year', 'Sentiment']).size().reset_index(name='Count')

        # Criar o treemap com Plotly Express
        fig = px.treemap(sentiment_counts, path=['Year', 'Sentiment'], values='Count', width=700, height=800)
        fig.update_traces(root_color="white", selector=dict(type='treemap'))
        st.plotly_chart(fig)
        
        
        
        
        if tabela:
            
            # Aplicar a função de análise de sentimento ao campo "clean_Abstract"
            df['Sentiment'] = df['clean_Abstract'].apply(analyze_sentiment)

            # Contagem dos resultados da análise de sentimento por ano
            sentiment_counts = df.groupby(['Year', 'Sentiment']).size().reset_index(name='Count')

            # Criar tabelas para cada ano
            anos_unicos = df['Year'].unique()
            for ano in anos_unicos:
                st.write(f"Tabela de análise de sentimento para o ano {ano}")
                tabela_ano = sentiment_counts[sentiment_counts['Year'] == ano]
                st.write(tabela_ano)

            # Tabela geral de sentimento
            st.write("Total de sentimento")
            tabela_geral = sentiment_counts.groupby('Sentiment')['Count'].sum().reset_index()
            st.write(tabela_geral)
            

        
        
        if grafico:
            
            # Aplicar a função de análise de sentimento ao campo "clean_Abstract"
            df['Sentiment'] = df['clean_Abstract'].apply(analyze_sentiment)

            # Contagem dos resultados da análise de sentimento por ano
            sentiment_counts = df.groupby(['Year', 'Sentiment']).size().reset_index(name='Count')

            # Criar um gráfico de bolhas com Plotly Express
            fig = px.scatter(sentiment_counts, x='Year', y='Sentiment', size='Count', color='Sentiment',
                            hover_name='Sentiment', hover_data={'Year': True, 'Sentiment': False, 'Count': True},
                            width=900, height=600)

            # Atualizar a aparência do gráfico de bolhas
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

            # Exibir o gráfico de bolhas no Streamlit
            st.plotly_chart(fig)
            
            
            
            
            
            
            
            
            
            
            
    if modificador == "Classificação Gramatical do Conteúdo por Ano":
        
        import pandas as pd
        import plotly.express as px
        from nltk import pos_tag, word_tokenize

        # Estrutura de dados fictícios para ilustrar o processo
        data = df

        df = pd.DataFrame(data)

        # Obtendo as POS tags de cada ano
        pos_tags_by_year = {}
        anos_unicos = df['Year'].unique()

        for ano in anos_unicos:
            data_year = df[df['Year'] == ano]
            all_pos_tags = [tag for sublist in data_year['clean_Abstract'].apply(lambda x: pos_tag(word_tokenize(x))) for _, tag in sublist]
            pos_counts = pd.Series(all_pos_tags).value_counts().reset_index()
            pos_counts.columns = ['POS', 'Count']
            pos_tags_by_year[ano] = pos_counts

        # Criar um dicionário contendo todos os dados de POS tags
        all_pos_tags = pd.concat(pos_tags_by_year.values(), keys=pos_tags_by_year.keys())

        # Normalizar os valores da contagem para facilitar a visualização no treemap
        max_count = all_pos_tags['Count'].max()
        all_pos_tags['Normalized_Count'] = all_pos_tags['Count'] / max_count

        # Criar o treemap com Plotly Express
        fig = px.treemap(all_pos_tags.reset_index(), path=['level_0', 'POS'], values='Normalized_Count' , width=700, height=800)
        fig.update_traces(root_color="white", selector=dict(type='treemap'))
        st.plotly_chart(fig)



        
        
        if tabela:
            
            # Obtendo as POS tags de cada ano
            pos_tags_by_year = {}
            anos_unicos = df['Year'].unique()

            for ano in anos_unicos:
                data_year = df[df['Year'] == ano]
                all_pos_tags = [tag for sublist in data_year['clean_Abstract'].apply(lambda x: pos_tag(word_tokenize(x))) for _, tag in sublist]
                pos_counts = pd.Series(all_pos_tags).value_counts().reset_index()
                pos_counts.columns = ['POS', 'Count']
                pos_tags_by_year[ano] = pos_counts

            # Criar tabelas separadas para cada ano
            for ano, pos_data in pos_tags_by_year.items():
                st.write(f"Tabela para o ano {ano}")
                st.write(pos_data)

            # Criar uma tabela final com a contagem total das POS tags
            all_pos_tags = pd.concat(pos_tags_by_year.values())
            total_pos_counts = all_pos_tags.groupby('POS')['Count'].sum().reset_index()
            st.write("Total das POS tags")
            st.write(total_pos_counts)
            
            
            
            
            
        if grafico:
            
            # Obtendo as POS tags de cada ano
            pos_tags_by_year = {}
            anos_unicos = df['Year'].unique()

            for ano in anos_unicos:
                data_year = df[df['Year'] == ano]
                all_pos_tags = [tag for sublist in data_year['clean_Abstract'].apply(lambda x: pos_tag(word_tokenize(x))) for _, tag in sublist]
                pos_counts = pd.Series(all_pos_tags).value_counts().reset_index()
                pos_counts.columns = ['POS', 'Count']
                pos_tags_by_year[ano] = pos_counts

            # Criar um DataFrame para o gráfico de bolhas
            df_bolhas = pd.concat(pos_tags_by_year.values())

            # Criar o gráfico de bolhas com Plotly Express
            fig = px.scatter(df_bolhas, x='POS', y='Count', size='Count', color='POS',
                            hover_name='POS', hover_data={'POS': False, 'Count': True},
                            width=900, height=600)

            # Atualizando a aparência do gráfico de bolhas
            fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))

            # Exibindo o gráfico de bolhas no Streamlit
            st.plotly_chart(fig)
                
    
    
    
    
    
    
    
    
    # fazer para a covid

    if modificador == "Influênia da Covid na Produção do Conhecimento por Ano":
        
        st.write("---")
        
        start, end = st.select_slider(
            'Selecione o intervalo temporal que deseja visualizar',
            options=['Pré-Covid 19', 'Início da Covid 19', 'Fim da Covid-19', 'Pós-Covid 19'],
            value=('Pré-Covid 19', 'Pós-Covid 19'))  

        st.write("---")
            
            
            
            
        if start == "Pré-Covid 19" and end == "Início da Covid 19":
            
            # Filtrar os dados entre os anos 2014 e 2019
            filtered_data = df[(df['Year'] >= 2014) & (df['Year'] <= 2019)]













# botões em baixo
with st.container():
    import webbrowser

    # Fazer por colunas
    st.write("---")
    st.subheader("Consulte os dados da pesquisa")

    col1, col2, col3 = st.columns(3)


    with col1:  # Botão do consulte dados da pesquisa
        btn = st.button("Ver dados através da SCOPUS")
        if btn:
            webbrowser.open_new_tab("https://www.scopus.com/")




    with col2:  # Botão para download do excel
        
        data = pd.DataFrame({
        'scopus.csv'
        })

        @st.cache_data
        def convert_df_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df_to_csv(data)

        
        st.download_button(
            label='Descarregar ficheiro CSV',
            data=csv,
            file_name='scopus.csv',
            mime='text/csv',
            )