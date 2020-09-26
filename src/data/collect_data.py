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


def initiate_vocab(test=True, test_name='test'):
    """ Initiate the vocab table

    Parameters
    -----
    test: boolean
        Is it a test
    test_name:
        name of the test

    Return
    -----
    vocab: pd.DataFrame
        The vocab table
    """

    if test:
        vocab = pd.read_csv(f'data/raw/german_english_{test_name}.csv')
    else:
        vocab = pd.read_csv('data/raw/german_english.csv')

    vocab['try_session_german_english'] = False
    vocab['try_session_english_german'] = False

    retry_german_english = vocab['retry_german_english'] < date.today().strftime("%Y-%m-%d")
    vocab.loc[retry_german_english, 'score_german_english'] -= 1
    vocab.loc[retry_german_english, 'retry_german_english'] = "2022-01-01"

    retry_english_german = vocab['retry_english_german'] < date.today().strftime("%Y-%m-%d")
    vocab.loc[retry_english_german, 'score_english_german'] -= 1
    vocab.loc[retry_english_german, 'retry_english_german'] = "2022-01-01"

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
    nb_known_words_session = 0
    return historical_data, triesSession, nb_known_words_session


def choose_a_language(vocab):
    """ Choose a language between German and English

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table

    Return
    -----
    input_language: str
        The input language
    output_language: str
        The output language
    """

    languages = ['german', 'english']
    possible_languages = []

    forbidden_words = (
            vocab['try_session_german_english'] |
            vocab['try_session_english_german'] |
            (vocab['score_german_english'] >= 5)
    )
    if len(vocab[forbidden_words]) != len(vocab):
        possible_languages.append('english')

    forbidden_words = (
            vocab['try_session_german_english'] |
            vocab['try_session_english_german'] |
            (vocab['score_english_german'] >= 5)
    )
    if len(vocab[forbidden_words]) != len(vocab):
        possible_languages.append('german')

    if not possible_languages:
        print("There are no more word for today!")
        return -1, -1

    output_language = random.choice(possible_languages)
    input_language = [x for x in languages if x != output_language][0]

    return input_language, output_language


def choose_a_word(vocab, input_language, output_language):
    """ Choose a word in vocab, which has not been successful in this session

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    input_language: str
        The input language
    output_language: str
        The output language

    Return
    -----
    id_vocab: int
        The word id
    """

    forbidden_words = (
            vocab['try_session_german_english'] |
            vocab['try_session_english_german'] |
            (vocab[f'score_{input_language}_{output_language}'] >= 5)
    )

    # Pick a random word
    id_vocab = random.choice(vocab[~forbidden_words]['id_vocab'].tolist())
    return id_vocab


def prompt_question(vocab, id_vocab, input_language, output_language, nb_known_words_session, tries_session):
    """ Prompt the word and language to find. Return the answer

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    id_vocab: int
        The word id of the chosen word
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
        output_language, vocab.loc[vocab['id_vocab'] == id_vocab, input_language].values[0])

    pre_question_time = datetime.datetime.now()
    your_guess = input(text_prompt)
    post_question_time = datetime.datetime.now()

    question_time = post_question_time - pre_question_time

    return your_guess, question_time


def if_correct():
    """Print statement if correct answer
    """
    print(f"{PrintColors.GREEN}Correct!{PrintColors.ENDC}")


def if_not_correct(vocab, id_vocab, input_language, output_language, your_guess, is_it_another_word):
    """Print statement + rewrite input if wrong answer

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    id_vocab: int
        The word id of the chosen word
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
            vocab.loc[vocab['id_vocab'] == id_vocab, input_language].values[0],
            vocab.loc[vocab['id_vocab'] == id_vocab, output_language].values[0]
        ))

    if is_it_another_word:
        print("{}You confused it{} with '{} = {}'".format(
            PrintColors.WARNING, PrintColors.ENDC,
            your_guess, vocab[vocab[output_language] == your_guess].iloc[0][input_language]))

    write_it_again = input("Write it again   ")
    return write_it_again


def add_historical_data(
        historical_data, vocab,
        id_vocab, input_language, output_language,
        is_it_correct, your_guess, question_time, write_it_again):
    """ Add try to historical data

    Parameters
    -----
    historical_data: pd.DataFrame()
        Historical data of the session
    vocab: pd.DataFrame
        The vocab table
    id_vocab: int
        The word id of the chosen word
    input_language: str
        The input language
    output_language: str
        The output language
    is_it_correct: boolean
        Is the answer correct
    your_guess: str
        The guess word
    question_time: datetime.timedelta
        Time it took to write an answer
    write_it_again: str
        What was written when it asked to write it again

    Return
    -----
    historicalData: pd.DataFrame()
        Historical data of the session

    """

    historical_data = historical_data.append({
        'id_vocab': vocab.loc[vocab['id_vocab'] == id_vocab, 'id_vocab'].values[0],
        'german_word': vocab.loc[vocab['id_vocab'] == id_vocab, 'german'].values[0],
        'english_word': vocab.loc[vocab['id_vocab'] == id_vocab, 'english'].values[0],
        'score_before': vocab.loc[vocab['id_vocab'] == id_vocab, f'score_{input_language}_{output_language}'].values[0],
        'score_before_other_language': vocab.loc[vocab['id_vocab'] == id_vocab, f'score_{output_language}_{input_language}'].values[0],
        'language_asked': output_language,
        'result': is_it_correct,
        'guess': your_guess,
        'question_time': question_time.seconds,
        "write_it_again": write_it_again,
        'datetime': datetime.datetime.now()
    }, ignore_index=True)

    return historical_data


def update_vocab(vocab, id_vocab, input_language, output_language, is_it_correct):
    """ Update the vocab table

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    id_vocab: int
        The word if of the chosen word
    input_language: str
        The input language
    output_language: str
        The output language
    is_it_correct: boolean
        Is the answer correct

    Return
    -----
    vocab: pd.DataFrame
        The vocab table

    """

    if is_it_correct:
        vocab.loc[
            vocab["id_vocab"] == id_vocab, f'score_{input_language}_{output_language}'
        ] += 1
    if not is_it_correct:
        vocab.loc[
            vocab["id_vocab"] == id_vocab, f'score_{input_language}_{output_language}'
        ] -= 1

    vocab.loc[vocab["id_vocab"] == id_vocab, f'try_session_{input_language}_{output_language}'] = True

    if vocab.loc[vocab["id_vocab"] == id_vocab, f'score_{input_language}_{output_language}'].values[0] == 5:
        print(f"{PrintColors.BOLDGREEN}Archived!!{PrintColors.ENDC}")

    return vocab


def update_historical_data(historical_data, test=True, test_name='test'):
    """ Update and save the historical data

    Parameters
    -----
    historical_data: pd.DataFrame()
        Historical data of the session
    test: boolean
        Is it a test
    test_name:
        name of the test

    Return
    -----
    None

    """

    if test:
        historical_data_old = pd.read_csv(f'data/raw/historical_data_{test_name}.csv')
    else:
        historical_data_old = pd.read_csv('data/raw/historical_data.csv')

    historical_data_new = pd.concat([historical_data_old, historical_data], axis=0)

    if test:
        historical_data_new.to_csv(f'data/raw/historical_data_{test_name}_after.csv', index=False)
    else:
        historical_data_new.to_csv('data/raw/historical_data.csv', index=False)


def finalize_vocab(vocab, test=True, test_name='test'):
    """ Update and save the vocab table

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    test: boolean
        Is it a test
    test_name:
        name of the test

    Return
    -----
    None

    """

    vocab.loc[vocab['score_german_english'] < -10, 'score_german_english'] = -10
    vocab.loc[vocab['score_english_german'] < -10, 'score_english_german'] = -10

    vocab.loc[
        (vocab['score_german_english'] == 5) & vocab['try_session_german_english'], 'retry_german_english'
    ] = (date.today() + datetime.timedelta(days=90)).strftime("%Y-%m-%d")
    vocab.loc[
        (vocab['score_english_german'] == 5) & vocab['try_session_english_german'], 'retry_english_german'
    ] = (date.today() + datetime.timedelta(days=90)).strftime("%Y-%m-%d")

    if test:
        vocab.to_csv(f'data/raw/german_english_{test_name}_after.csv', index=False)
    else:
        vocab.to_csv('data/raw/german_english.csv', index=False)
