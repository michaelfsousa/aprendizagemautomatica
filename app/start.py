# -*- coding: utf-8 -*-
import serial, os, signals, sys, suggestions
from sklearn.externals import joblib

'''
Este módulo é usado para registrar novos sinais,
e testar as previsões.

Você pode usar modos diferentes para obter coisas diferentes:

TARGET MODE: É usado para registrar novas amostras
Você deve especificar o sinal de destino e o lote com esta sintaxe:

python start.py target=a:3

Onde 'a' é o sinal de destino e 3 é o lote.
Ele salva o sinal gravado em um arquivo chamado "a_sample_3_N.txt" no
diretório especificado pelo "target_directory".
PS. N é um número progressivo no lote usado para tornar cada gravação única.

TARGET_ALL MODE: É usado para registrar novas amostras, fornecendo uma sentença
que contém todas as letras. O usuário tem que escrever a frase usando o
dispositivo e o módulo salva a amostra gravada correspondente.

WRITE MODE: Registra novas amostras e traduza-as em texto, prevendo
o caractere correto. Se o parâmetro "noautocorrect" for usado,
as previsões não serão corrigidas de forma cruzada com o dicionário.

Ao prever um dos caracteres maiusculos abaixo, ocorrerá certas ações: 

D = deleta o último caractere escrito
A = deleta todo o texto 

Essas ações podem ser habilitadas alterando para True a variável DELETE_ALL_ENABLED
'''

def print_sentence_with_pointer(sentence, position):
    print(sentence)
    print(" "*position + "^")

#Sentença usada para obter amostras porque contém todas as letras(pantograma).
#ALTERNATIVA EM INGLÊS: the quick brown fox jumps over the lazy dog
#ALTERNATIVA EM PORTUGUÊS: tv faz quengo explodir com whisky jb
#ALTERNATIVA EM PORTUGUÊS INCLUINDO TODAS AS LETRAS ACENTUADAS: À noite, vovô Kowalsky vê o ímã cair no pé do pinguim queixoso e vovó põe açúcar no chá de tâmaras do jabuti feliz.
test_sentence = "tv faz quengo explodir com whisky jb"

# Modo de parâmetros, que podem ser controlados usando sys.argv pelo terminal
TRY_TO_PREDICT = False
SAVE_NEW_SAMPLES = False
FULL_CYCLE = False
ENABLE_WRITE = False
TARGET_ALL_MODE = False
AUTOCORRECT = False
DELETE_ALL_ENABLED = True


#Parâmetros da porta serial
SERIAL_PORT = "COM8"
BAUD_RATE = 115200
TIMEOUT = 100

#Parâmetros de gravação
target_sign = "a"
current_batch = "0"
target_directory = "data"
current_test_index = 0

#Analisa os argumentos para ativar um modo específico
arguments = {}

for i in sys.argv[1:]:
    if "=" in i:
        sub_args = i.split("=")
        arguments[sub_args[0]]=sub_args[1]
    else:
        arguments[i]=None

#Se houver argumentos, será analisado
if len(sys.argv)>1:
    if "target" in arguments:
        target_sign = arguments["target"].split(":")[0]
        current_batch = arguments["target"].split(":")[1]
        print("Sinal alvo: '{sign}' usando o lote: {batch}".format(sign=target_sign, batch=current_batch))
        SAVE_NEW_SAMPLES = True
    if "predict" in arguments:
        TRY_TO_PREDICT = True
    if "write" in arguments:
        TRY_TO_PREDICT = True
        ENABLE_WRITE = True
    if "test" in arguments:
        current_batch = arguments["test"]
        TARGET_ALL_MODE = True
        SAVE_NEW_SAMPLES = True
    if "autocorrect" in arguments:
        AUTOCORRECT=True
    if "port" in arguments:
        SERIAL_PORT = arguments["port"]

clf = None
classes = None
sentence = ""

hinter = suggestions.Hinter.load_portuguese_dict()

#Se preddict foi mencionado no terminal, carrega o modelo de aprendizado de máquina do arquivo, bem como as classes
if TRY_TO_PREDICT:
    print("Carregando modelo e classes...")
    clf = joblib.load('model.pkl')
    classes = joblib.load('classes.pkl')


print("Porta serial '{port}' aberta com baudrate {baud}...".format(port=SERIAL_PORT, baud=BAUD_RATE))

print("Para finalizar o programa, pressione o botão 'Reset' do dispositivo ")

#Abre a porta serial com a porta e baudrate específicos nas variáveis
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)

output = []

in_loop = True
is_recording = False

current_sample = 0

#Reinicia o arquivo de saída 'output.txt'
output_file = open("output.txt","w")
output_file.write("")
output_file.close()

# Se TARGET_ALL_MODE = True, imprime as sentenças com a posição corrente
if TARGET_ALL_MODE:
    print_sentence_with_pointer(test_sentence, 0)

try:
    while in_loop:
        #Ler uma linha através da porta serial e apaga os terminadores de linha
        lin = ser.readline()
        line = lin.decode("utf-8")

        line = line.replace("\r\n", "")
        print(line),
        # Se receber "STARTING BATCH" da porta serial é iniciado a gravação
        if line == "STARTING BATCH":
            #habilita a gravação
            is_recording = True
            #Reinicia o buffer
            output = []
            print("GRAVANDO..."),
        elif line == "CLOSING BATCH": #Pára a gravação e analisa o resultado
            #Desabilita a gravação
            is_recording = False
            if len(output)>1: #Se for menor que 1, significa erro
                print("CONCLUÍDO! SALVANDO..."),

                #Se TARGET_ALL_MODE = True, é alterado o sinal de destino de acordo com a posição
                if TARGET_ALL_MODE:
                    if current_test_index<len(test_sentence):
                        target_sign = test_sentence[current_test_index]
                    else:
                        #No final da frase, sai do programa
                        print("Todos os alvos finalizados")
                        quit()

                #Gera o nome do arquivo com base no sinal de destino, lote e número progressivo
                filename = "{sign}_sample_{batch}_{number}.txt".format(sign=target_sign, batch=current_batch, number=current_sample)
                #Gera o caminho
                path = target_directory + os.sep + filename

                #Se SAVE_NEW_SAMPLES == False, salva a gravação em um arquivo temporário
                if SAVE_NEW_SAMPLES == False:
                    path = "tmp.txt"
                    filename = "tmp.txt"

                #Salva a gravação em um arquivo
                f = open(path, "w")
                f.write('\n'.join(output))
                f.close()
                print("SALVO EM {filename}".format(filename=filename))
                current_sample += 1

                # Se TRY_TO_PREDICT == True, utiliza o modelo para prever a gravação
                if TRY_TO_PREDICT:
                    print("PREVENDO...")
                    # Carrega a gravação como um objeto Sample
                    sample_test = signals.Sample.load_from_file(path)
                    linearized_sample = sample_test.get_linearized(reshape=True)
                    # Prevê o número com o modelo de aprendizado de máquina
                    number = clf.predict(linearized_sample)
                    #Convertê-lo para um char
                    char = chr(ord('a')+number[0])

                    #Obtém a última palavra na frase
                    last_word = sentence.split(" ")[-1:][0]


                    #Se AUTOCORRECT == True, a palavra encontrada pode ser substituída por um elemento por cross-calculçated
                    # contido no dicionário
                    if AUTOCORRECT and char.islower():
                        predicted_char = hinter.most_probable_letter(clf, classes, linearized_sample, last_word)
                        if predicted_char is not None:
                            print("PALAVRA ATUAL: {word}, PREVISTO: {old}, COM CROSS_CALCULATED: {new}".format(word=last_word, old=char, new=predicted_char))
                            char = predicted_char

                    # Se o modo for WRITE, atribui significados especiais a alguns caracteres
                    # e cria uma frase com cada caractere
                    if ENABLE_WRITE:
                        if char == 'D': #Deleta o último caractere
                            sentence = sentence[:-1]
                        elif char == 'A': #Deleta todos os caracteres
                            if DELETE_ALL_ENABLED:
                                sentence = ""
                            else:
                                print("DELETE_ALL_ENABLED = FALSE")
                        else: #Adiciona o char na setença
                            sentence += char
                        #Imprime o último caractere e a sentença
                        print ("[{char}] -> {sentence}".format(char=char, sentence=sentence))

                        #Salva a saída num arquivo
                        output_file = open("output.txt","w")
                        output_file.write(sentence)
                        output_file.close()
                    else:
                        print(char)
            else: #No caso de uma sequÊncia corrompida por algum motivo(poucio conteúdo na sample gerada...
                print("OCORREU UM ERRO...")
                current_test_index -= 1

            #Se TARGET_ALL_MODE=True mostra a posição atual na sentença
            if TARGET_ALL_MODE:
                current_test_index += 1
                print_sentence_with_pointer(test_sentence, current_test_index)
        else:
            #Anexa a linha de sinal atual na gravação
            output.append(line)
except KeyboardInterrupt: #Quando CTRL+C é pressionado, o loop termina
    print('LOOP FECHADO!')

#Fecha a porta serial
ser.close()
