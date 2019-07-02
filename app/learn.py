from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import signals
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV

'''
Este módulo treina o algoritmo de aprendizado de máquina e salva o modelo
em "model.pkl".
Também salva as classes em um arquivo "classes.pkl"

Analisa o conjunto de dados contido no diretório "data".
O conjunto de dados será composto de N arquivos. Cada um desses arquivos é
uma gravação de um gesto específico.
O nome do arquivo representa o significado da gravação.
Por exemplo, o arquivo:
a_sample_0_1.txt
É uma gravação para o sinal "a", gravado no lote "0".

'''
#Verifica se o módulo é executado como principal, necessário para processamento paralelo
if __name__ == '__main__':
    #Lista de parâmetros
    SHOW_CONFUSION_MATRIX = True

    x_data = []
    y_data = []

    classes = {}

    root = "data" #Diretório padrão contendo o conjunto de dados

    print("Carregando o conjunto de dados de '{directory}'...".format(directory=root)),

    # Buscar todos os arquivos de dados do diretório raiz do dataset
    for path, subdirs, files in os.walk(root):
        for name in files:
            #Retorna o nome do arquivo
            filename = os.path.join(path, name)
            #Carrega a amostra do arquivo
            sample = signals.Sample.load_from_file(filename)
            #Lineariza a amostra e, em seguida, adiciona à lista x_data
            x_data.append(sample.get_linearized())
            #Extrai a categoria do nome do arquivo
            #Por exemplo, o arquivo "a_sample_0.txt" será considerado como "a"
            category = name.split("_")[0]
            # Obtém um número para a categoria, como um deslocamento da categoria
            # para o char em Ascii
            number = ord(category) - ord("a")
            # Adiciona a categoria à lista y_data
            y_data.append(number)
            # Inclui a categoria e o número correspondente em um dicionário
            # para facilitar o acesso e a referência
            classes[number] = category

    print("CONCLUÍDO")

    # Parâmetros usados ​​no processo de treinamento validado de forma cruzada
    # A biblioteca tenta automaticamente todas as combinações possíveis
    # encontrar a melhor pontuação.
    params = {'C': [0.001, 0.01, 0.1, 1], 'kernel': ['linear']}

    #Inicializa o modelo
    svc = svm.SVC(probability=True)
    # Inicializa o GridSearchCV com 8 núcleos de processamento e a máxima verbosidade
    clf = GridSearchCV(svc, params, verbose=10, n_jobs=8)

    # Divide o conjunto de dados em dois subconjuntos, um usado para treinamento e outro para teste
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.35, random_state=0)

    print("Iniciando o processo de treinamento...")

    #Inicia o processo de treinamento
    clf.fit(X_train, Y_train)

    #Se SHOW_CONFUSION_MATRIX é verdadeiro, imprime a matriz de confusão
    if SHOW_CONFUSION_MATRIX:
        print("Matriz de confusão:")
        Y_predicted = clf.predict(X_test)
        print(confusion_matrix(Y_test, Y_predicted))

    print("\nMelhores parâmetros estimados: ")
    print(clf.best_estimator_)

    #Calculates the score of the best estimator found.
    score = clf.score(X_test, Y_test)

    print("\nSCORE: {score}\n".format(score=score))

    print("Salvando o modelo..."),

    #Salva o modelo para o arquivo "model.pkl"
    joblib.dump(clf, 'model.pkl')
    #Salva as classes para o arquivo "classes.pkl"
    joblib.dump(classes, 'classes.pkl')

    print ("CONCLUIDO!!!!")