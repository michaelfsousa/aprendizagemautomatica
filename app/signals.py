# -*- coding: utf-8 -*-
import numpy as np
from sklearn.preprocessing import scale
from scipy.interpolate import interp1d

'''
Esta biblioteca contém as classes necessárias para processar os sinais.
'''

class Sample:
    '''
    A amostra é usada para carregar, armazenar e processar os sinais obtidos
    do acelerômetro.
    Ele fornece um método para carregar os sinais do arquivo e processá-los.
    '''
    def __init__(self, acx, acy, acz, gx, gy, gz):
        self.acx = acx
        self.acy = acy
        self.acz = acz
        self.gx = gx
        self.gy = gy
        self.gz = gz

    def get_linearized(self, reshape=False):
        '''
        Lineariza os dados, combinando os 6 eixos diferentes.
        Útil para alimentar os dados em um algoritmo de aprendizado de máquina.

        Se reshape = True, ele é reformulado (Útil ao alimentá-lo com o método de previsão)
        '''
        if reshape:
            return np.concatenate((self.acx, self.acy, self.acz, self.gx, self.gy, self.gz)).reshape(1,-1)
        else:
            return np.concatenate((self.acx, self.acy, self.acz, self.gx, self.gy, self.gz))


    @staticmethod
    def load_from_file(filename, size_fit = 50):
        '''
       Carrega os dados do sinal de um arquivo.

        filename: indica o caminho do arquivo.
        size_fit: é o número final de amostra que um axe terá.
                  Utiliza interpolação linear para aumentar ou diminuir
                  o número de amostras.

        '''
        # Carrega os dados do sinal do arquivo como uma lista
        # Ignora a primeira e a última linha e converte cada número em um int
        data_raw = [map(lambda x: int(x), i.split(" ")[1:-1]) for i in open(filename)]

        #(data = np.array(data_raw).astype(float))
        # Como não é possível converter diretamente o mapa de arquivos em float pelo numpy np pyhon 3.x(relatado o problema),
        # faz-se necessário listar cada mapa,
        # convertendo-as individualmente e, assim, converter pelo float do numpy
        # lista cada mapa, convertendo-as em float cada uma
        dat = []
        for l in data_raw:
            dat.append([float(i) for i in l])
        # Converte os dados em float pelo numpy
        # data = np.array(dat).astype(float)
        data = np.array(dat, dtype=float)
        print(data)
        # Padroniza os dados dimensionando-os
        data_norm = scale(data)
        print("normalização dos dados")
        print(data_norm)
        # Extrai cada axe em uma variável separada
        # Esses apresentam a aceleração nos 3 eixos
        acx = data_norm[:,0]
        acy = data_norm[:,1]
        acz = data_norm[:,2]

        # Esses apresentam a rotação nos 3 eixos
        gx = data_norm[:,3]
        gy = data_norm[:,4]
        gz = data_norm[:,5]

        # Cria uma função para cada axe que interpola as amostras
        x = np.linspace(0, data.shape[0], data.shape[0])
        f_acx = interp1d(x, acx)
        f_acy = interp1d(x, acy)
        f_acz = interp1d(x, acz)

        f_gx = interp1d(x, gx)
        f_gy = interp1d(x, gy)
        f_gz = interp1d(x, gz)

        # Cria um novo conjunto de amostras com o tamanho de amostra desejado por redimensionamento do original

        xnew = np.linspace(0, data.shape[0], size_fit)
        acx_stretch = f_acx(xnew)
        acy_stretch = f_acy(xnew)
        acz_stretch = f_acz(xnew)

        gx_stretch = f_gx(xnew)
        gy_stretch = f_gy(xnew)
        gz_stretch = f_gz(xnew)
        # Retorna uma amostra com os valores calculados
        return Sample(acx_stretch, acy_stretch, acz_stretch, gx_stretch, gy_stretch, gz_stretch)
