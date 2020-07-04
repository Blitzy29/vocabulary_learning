import numpy as np
import pandas as pd
import random
import datetime
from datetime import date


class PrintColors:
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
    BOLDGREEN = '\033[1;92m'


def initiate_vocab():
    """ Initiate the vocab table

    Parameters
    -----

    Return
    -----
    vocab: pd.DataFrame
        The vocab table
    """

    vocab = pd.read_csv('data/raw/german_english.csv')

    vocab['succeed_session'] = False
    vocab['try_session'] = False

    vocab['score'] = np.where(vocab['retry'] < date.today().strftime("%Y-%m-%d"), vocab['score'] - 1, vocab['score'])
    vocab['retry'] = np.where(vocab['retry'] < date.today().strftime("%Y-%m-%d"), "2022-01-01", vocab['retry'])
    vocab['succeed'] = np.where(vocab['retry'] < date.today().strftime("%Y-%m-%d"), False, vocab['succeed'])

    return vocab


def initiate_session():
    """ Initiate the session info

    Parameters
    -----

    Return
    -----
    historical_data: pd.DataFrame()
        Historical data of the session
    triesSession: int
        Number of tries during this session
    """

    historical_data = pd.DataFrame()
    triesSession = 0
    return historical_data, triesSession


def choose_a_word(vocab):
    """ Choose a word in vocab, which has not been successful in this session

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table

    Return
    -----
    row_to_test: int
        The row index of the chosen word
    """

    word_found = False
    while not word_found:
        # Pick a random word
        row_to_test = np.random.randint(0, len(vocab))

        if vocab.loc[row_to_test, 'succeed_session'] | vocab.loc[row_to_test, 'succeed']:
            continue
        else:
            return row_to_test


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

    possible_actions = ['german', 'english']
    output_language = random.choice(possible_actions)
    input_language = [x for x in possible_actions if x != output_language][0]
    return input_language, output_language


def prompt_question(vocab, row_to_test, input_language, output_language, nb_known_words_session, tries_session):
    """ Prompt the word and language to find. Return the answer

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    row_to_test: int
        The row index of the chosen word
    input_language: str
        The input language
    output_language: str
        The output language
    nb_known_words_session: int
        Number of words known during this session
    tries_session: int
        Number of tries during this session

    Return
    -----
    your_guess: str
        The answer
    """

    text_prompt = "{}{}/{}{} - What is the {} word for '{}'?   ".format(
        PrintColors.BOLD, nb_known_words_session, tries_session, PrintColors.ENDC,
        output_language, vocab.loc[row_to_test, input_language])
    your_guess = input(text_prompt)
    return your_guess


def if_correct():
    """Print statement if correct answer
    """
    print(f"{PrintColors.GREEN}Correct!{PrintColors.ENDC}")


def if_not_correct(vocab, row_to_test, input_language, output_language, your_guess, is_it_another_word):
    """Print statement + rewrite input if wrong answer

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    row_to_test: int
        The row index of the chosen word
    input_language: str
        The input language
    output_language: str
        The output language
    your_guess: str
        The answer
    is_it_another_word: boolean
        Does the answer correspond to another word

    Return
    -----
    None

    """

    print(
        "{}Sorry{}, it was '{} = {}'".format(
            PrintColors.FAIL, PrintColors.ENDC,
            vocab.loc[row_to_test, input_language],
            vocab.loc[row_to_test, output_language]
        ))

    if is_it_another_word:
        print("{}You confused it{} with '{} = {}'".format(
            PrintColors.WARNING, PrintColors.ENDC,
            your_guess, vocab[vocab[output_language] == your_guess].iloc[0][input_language]))

    input("Write it again   ")


def add_historical_data(historical_data, vocab, row_to_test, output_language, is_it_correct):
    """ Add try to historical data

    Parameters
    -----
    historical_data: pd.DataFrame()
        Historical data of the session
    vocab: pd.DataFrame
        The vocab table
    row_to_test: int
        The row index of the chosen word
    output_language: str
        The output language
    is_it_correct: boolean
        Is the answer correct

    Return
    -----
    historicalData: pd.DataFrame()
        Historical data of the session

    """

    historical_data = historical_data.append({
        'german_word': vocab.loc[row_to_test, 'german'],
        'english_word': vocab.loc[row_to_test, 'english'],
        'score_before': vocab.loc[row_to_test, 'score'],
        'language_asked': output_language,
        'result': is_it_correct
    }, ignore_index=True)

    return historical_data


def update_vocab(vocab, row_to_test, is_it_correct):
    """ Update the vocab table

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    row_to_test: int
        The row index of the chosen word
    is_it_correct: boolean
        Is the answer correct

    Return
    -----
    vocab: pd.DataFrame
        The vocab table

    """

    vocab.loc[row_to_test, 'succeed_session'] = is_it_correct

    if is_it_correct:
        vocab.loc[row_to_test, 'score'] = vocab.loc[row_to_test, 'score'] + 1
    if not is_it_correct:
        vocab.loc[row_to_test, 'score'] = vocab.loc[row_to_test, 'score'] - 1

    vocab.loc[row_to_test, 'try_session'] = True

    if vocab.loc[row_to_test, 'score'] == 5:
        print(f"{PrintColors.BOLDGREEN}Archived!!{PrintColors.ENDC}")

    return vocab


def update_historical_data(historical_data):
    """ Update and save the historical data

    Parameters
    -----
    historical_data: pd.DataFrame()
        Historical data of the session

    Return
    -----
    None

    """

    historical_data['day'] = date.today().strftime("%Y-%m-%d")
    historical_data_old = pd.read_csv('data/raw/historical_data.csv')
    historical_data_new = pd.concat([historical_data_old, historical_data], axis=0)
    historical_data_new.to_csv('data/raw/historical_data.csv', index=False)


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

    vocab['score'] = np.where(vocab['score'] < -10, -10, vocab['score'])
    vocab['succeed'] = vocab['score'] >= 5
    vocab['retry'] = np.where(vocab['succeed'] & vocab['succeed_session'],
                              (date.today() + datetime.timedelta(days=90)).strftime("%Y-%m-%d"), vocab['retry'])

    vocab.to_csv('data/raw/german_english.csv', index=False)
