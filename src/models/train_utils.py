import numpy as np
import pandas as pd
import random
import datetime
from datetime import date


class bcolors:
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DARKCYAN = '\033[36m'
    FAIL = '\033[91m'
    RED = '\033[91m'
    OKGREEN = '\033[92m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[94m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    HEADER = '\033[95m'
    CYAN = '\033[96m'


def initiate_vocab():
    """ Initiate the vocab table

    Parameters
    -----

    Return
    -----
    vocab: pd.DataFrame
        The vocab table
    """

    vocab = pd.read_csv('data/processed/GermanEnglish.csv')

    vocab['SucceedSession'] = False
    vocab['TrySession'] = False

    vocab['Score'] = np.where(vocab['Retry'] < date.today().strftime("%Y-%m-%d"), vocab['Score'] - 1, vocab['Score'])
    vocab['Retry'] = np.where(vocab['Retry'] < date.today().strftime("%Y-%m-%d"), "2022-01-01", vocab['Retry'])
    vocab['Succeed'] = np.where(vocab['Retry'] < date.today().strftime("%Y-%m-%d"), False, vocab['Succeed'])

    return vocab


def initiate_session():
    """ Initiate the session info

    Parameters
    -----

    Return
    -----
    historicalData: pd.DataFrame()
        Historical data of the session
    triesSession: int
        Number of tries during this session
    """

    historicalData = pd.DataFrame()
    triesSession = 0
    return historicalData, triesSession


def choose_a_word(vocab):
    """ Choose a word in vocab, which has not been successful in this session

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table

    Return
    -----
    rowToTest: int
        The row index of the chosen word
    """

    word_found = False
    while not word_found:
        # Pick a random word
        rowToTest = np.random.randint(0, len(vocab))

        if vocab.loc[rowToTest, 'SucceedSession'] | vocab.loc[rowToTest, 'Succeed']:
            continue
        else:
            return rowToTest


def choose_a_language():
    """ Choose a language between German and English

    Parameters
    -----

    Return
    -----
    input_language: str
        The input language
    output_language: str
        The output language
    """

    possible_actions = ['German', 'English']
    output_language = random.choice(possible_actions)
    input_language = [x for x in possible_actions if x != output_language][0]
    return input_language, output_language


def prompt_question(vocab, rowToTest, input_language, output_language, nKnownWordsSession, triesSession):
    """ Prompt the word and language to find. Return the answer

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    rowToTest: int
        The row index of the chosen word
    input_language: str
        The input language
    output_language: str
        The output language
    nKnownWordsSession: int
        Number of words known during this session
    triesSession: int
        Number of tries during this session

    Return
    -----
    yourGuess: str
        The answer
    """

    text_prompt = "{}{}/{}{} - What is the {} for '{}'?   ".format(
        bcolors.BOLD, nKnownWordsSession, triesSession, bcolors.ENDC,
        output_language, vocab.loc[rowToTest, input_language])
    yourGuess = input(text_prompt)
    return yourGuess


def if_correct():
    """Print statement if correct answer
    """
    print(f"{bcolors.OKGREEN}Correct!{bcolors.ENDC}")


def if_not_correct(vocab, rowToTest, input_language, output_language, yourGuess, isItAnotherWord):
    """Print statement + rewrite input if wrong answer

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    rowToTest: int
        The row index of the chosen word
    input_language: str
        The input language
    output_language: str
        The output language
    yourGuess: str
        The answer
    isItAnotherWord: boolean
        Does the answer correspond to another word

    Return
    -----
    None

    """

    print(
        "{}Sorry{}, it was '{} = {}'".format(
            bcolors.FAIL, bcolors.ENDC,
            vocab.loc[rowToTest, input_language],
            vocab.loc[rowToTest, output_language]
        ))

    if isItAnotherWord:
        print("{}You confused it{} with '{} = {}'".format(
            bcolors.WARNING, bcolors.ENDC,
            yourGuess, vocab[vocab[output_language] == yourGuess].iloc[0][input_language]))

    input("Write it again   ")


def add_historical_data(historicalData, vocab, rowToTest, output_language, isItCorrect):
    """ Add try to historical data

    Parameters
    -----
    historicalData: pd.DataFrame()
        Historical data of the session
    vocab: pd.DataFrame
        The vocab table
    rowToTest: int
        The row index of the chosen word
    output_language: str
        The output language
    isItCorrect: boolean
        Is the answer correct

    Return
    -----
    historicalData: pd.DataFrame()
        Historical data of the session

    """

    historicalData = historicalData.append({
        'germanWord': vocab.loc[rowToTest, 'German'],
        'englishWord': vocab.loc[rowToTest, 'English'],
        'scoreBefore': vocab.loc[rowToTest, 'Score'],
        'languageAsked': output_language,
        'result': isItCorrect
    }, ignore_index=True)

    return historicalData


def update_vocab(vocab, rowToTest, isItCorrect):
    """ Update the vocab table

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    rowToTest: int
        The row index of the chosen word
    isItCorrect: boolean
        Is the answer correct

    Return
    -----
    vocab: pd.DataFrame
        The vocab table

    """

    vocab.loc[rowToTest, 'SucceedSession'] = isItCorrect

    if isItCorrect:
        vocab.loc[rowToTest, 'Score'] = vocab.loc[rowToTest, 'Score'] + 1
    if not isItCorrect:
        vocab.loc[rowToTest, 'Score'] = vocab.loc[rowToTest, 'Score'] - 1

    vocab.loc[rowToTest, 'TrySession'] = True

    if vocab.loc[rowToTest, 'Score'] == 5:
        print("Archived!!")

    return vocab


def update_historical_data(historicalData):
    """ Update and save the historical data

    Parameters
    -----
    historicalData: pd.DataFrame()
        Historical data of the session

    Return
    -----
    None

    """

    historicalData['day'] = date.today().strftime("%Y-%m-%d")
    historicalDataOld = pd.read_csv('data/processed/HistoricalData.csv')
    historicalDataNew = pd.concat([historicalDataOld, historicalData], axis=0)
    historicalDataNew.to_csv('data/processed/HistoricalData.csv', index=False)


def finalize_vocab(vocab):
    """ Update and save the vocab table

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table

    Return
    -----
    None

    """

    vocab['Score'] = np.where(vocab['Score'] < -10, -10, vocab['Score'])
    vocab['Succeed'] = vocab['Score'] >= 5
    vocab['Retry'] = np.where(vocab['Succeed'] & vocab['SucceedSession'],
                              (date.today() + datetime.timedelta(days=90)).strftime("%Y-%m-%d"), vocab['Retry'])

    vocab.to_csv('data/processed/GermanEnglish.csv', index=False)
