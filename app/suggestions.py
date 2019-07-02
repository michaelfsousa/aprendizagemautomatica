# -*- coding: utf-8 -*-
import os

'''
Esta biblioteca contém as classes necessárias para manipular palavras e obter sugestões
'''

class Hinter:
    '''
    O Hinter é usado para carregar um dicionário e obter algumas sugestões
    sobre as próximas letras possíveis ou palavras compatíveis.
    '''
    def __init__(self, words):
        self.words = words

    @staticmethod
    def load_portuguese_dict():
        '''
        Carrega o dicionário em inglês e retorna um objeto Hinter com as palavras
        carregadas na lista self.words
        '''
        portuguese_filename = "dict" + os.sep + "PT_BR.txt"
        words = [i.replace("\n", "") for i in open(portuguese_filename, encoding="utf8")]
        return Hinter(words)

    def compatible_words(self, word, limit=10):
        '''
        Retorna as palavras que começam com o parâmetro "word".
        O parâmetro "limit" define quantas palavras a função
        retorna no máximo
        '''
        output = []
        word_count = 0
        # Percorra todas as palavras para encontrar as que começam com "word"
        for i in self.words:
            if i.startswith(word):
                output.append(i)
                word_count+=1
            if word_count>=limit: # Se o limite for atingido, saia
                break
        return output

    def next_letters(self, word):
        '''
        Retorna uma lista de letras compatíveis.
        Uma letra é compatível quando existem palavras que começam com "word"
        e são seguidos pela letra.
        '''
        # Procure 100 palavras compatíveis
        words = self.compatible_words(word, 100)
        letters = []
        # Percorra por todas as palavras compatíveis
        for i in words:
            if len(i)>len(word): # se a "word" for maior que uma palavra compatível, pule
                letter = i[len(word):len(word)+1] #Obtenha a próxima letra
                if not letter in letters: #Evitar duplicidades
                    letters.append(letter)
        return letters

    def does_word_exists(self, word):
        '''
        Verifica se existe uma palavra específica no dicionário carregado
        '''
        if word in self.words:
            return True
        else:
            return False

    def most_probable_letter(self, clf, classes, linearized_sample, word):
        '''
        Obtenha a letra mais provável para um determinado sinal gravado e a palavra atual
        '''
        if word=="":
            return None

        probabilities = clf.predict_log_proba(linearized_sample)
        ordered = sorted(classes)
        values = {}
        for i in range(len(probabilities[0])):
            values[round(probabilities[0,i], 5)] = classes[ordered[i]]
        ordered = sorted(values, reverse=True)
        possible_letters = self.next_letters(word)
        for i in ordered:
            if values[i] in possible_letters:
                return values[i]
        return None

