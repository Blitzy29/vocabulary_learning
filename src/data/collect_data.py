import numpy as np
import pandas as pd
import random
import datetime

from scipy.special import softmax

from src.visualization.visualize_vocab import plot_comparison_both_probabilities


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
        probas_next_session = pd.read_csv(f"data/raw/predictions_next_session_{test_name}.csv")
    else:
        vocab = pd.read_csv('data/official/german_english.csv')
        probas_next_session = pd.read_csv('data/official/predictions_next_session.csv')

    vocab['try_session_german_english'] = False
    vocab['try_session_english_german'] = False

    vocab = pd.merge(
        vocab,
        probas_next_session,
        on='id_vocab'
    )

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


def choose_a_language(i_vocab_try, vocab):
    """ Choose a language between German and English

    Parameters
    -----
    i_vocab_try: dict
        try
    vocab: pd.DataFrame
        The vocab table

    Return
    -----
    i_vocab_try: dict
        try
        input_language: str
            The input language
        output_language: str
            The output language
    """

    languages = ['german', 'english']
    possible_languages = []

    forbidden_words = (
            vocab['try_session_german_english'] |
            vocab['try_session_english_german']
    )
    if len(vocab[forbidden_words]) != len(vocab):
        possible_languages.append('english')

    forbidden_words = (
            vocab['try_session_german_english'] |
            vocab['try_session_english_german']
    )
    if len(vocab[forbidden_words]) != len(vocab):
        possible_languages.append('german')

    if not possible_languages:
        print("There are no more word for today!")
        i_vocab_try['input_language'] = -1
        i_vocab_try['output_language'] = -1
        return i_vocab_try

    output_language = random.choice(possible_languages)
    input_language = [x for x in languages if x != output_language][0]

    i_vocab_try['input_language'] = input_language
    i_vocab_try['output_language'] = output_language

    return i_vocab_try


def choose_a_word(i_vocab_try, vocab):
    """ Choose a word in vocab, which has not been successful in this session

    Parameters
    -----
    i_vocab_try: dict
        try
    vocab: pd.DataFrame
        The vocab table

    Return
    -----
    i_vocab_try:
        try
        id_vocab: int
            The word id
    """

    forbidden_words = (
            vocab['try_session_german_english'] |
            vocab['try_session_english_german']
    )

    list_possible_words = vocab[~forbidden_words]

    list_possible_words[
        f"{i_vocab_try['output_language']}_softmax"
    ] = softmax(3*(1 - list_possible_words[f"{i_vocab_try['output_language']}_proba"]))

    # Pick a random word
    id_vocab = np.random.choice(
        list_possible_words['id_vocab'].tolist(),
        p=list_possible_words[f"{i_vocab_try['output_language']}_softmax"]
    )

    i_vocab_try['id_vocab'] = id_vocab

    i_vocab_try['german_word'] = vocab.loc[
        vocab['id_vocab'] == id_vocab, 'german'
    ].values[0]

    i_vocab_try['english_word'] = vocab.loc[
        vocab['id_vocab'] == id_vocab, 'english'
    ].values[0]

    i_vocab_try['word_input_language'] = vocab.loc[
        vocab['id_vocab'] == id_vocab, i_vocab_try['input_language']
    ].values[0]

    i_vocab_try['word_output_language'] = vocab.loc[
        vocab['id_vocab'] == id_vocab, i_vocab_try['output_language']
    ].values[0]

    i_vocab_try['score_before'] = vocab.loc[
        vocab['id_vocab'] == id_vocab, f"score_{i_vocab_try['input_language']}_{i_vocab_try['output_language']}"
    ].values[0]

    i_vocab_try['score_before_other_language'] = vocab.loc[
        vocab['id_vocab'] == id_vocab, f"score_{i_vocab_try['output_language']}_{i_vocab_try['input_language']}"
    ].values[0]

    return i_vocab_try


def prompt_question(i_vocab_try, print_info):
    """ Prompt the word and language to find. Return the answer

    Parameters
    -----
    i_vocab_try: dict
        try
        id_vocab: int
            The word id of the chosen word
        input_language: str
            The input language
        output_language: str
            The output language
    print_info: dict()
        nb_known_words_session: int
            Number of words known during this session
        tries_session: int
            Number of tries during this session

    Return
    -----
    i_vocab_try: dict()
        try
        your_guess: str
            The answer
        question_time
    """

    text_prompt = "{}{}/{}{} - What is the {} word for '{}'?   ".format(
        PrintColors.BOLD, print_info['nb_known_words_session'], print_info['tries_session'], PrintColors.ENDC,
        i_vocab_try['output_language'], i_vocab_try['word_input_language'])

    i_vocab_try['datetime'] = datetime.datetime.now()

    pre_question_time = datetime.datetime.now()
    your_guess = input(text_prompt)
    post_question_time = datetime.datetime.now()

    question_time = post_question_time - pre_question_time

    i_vocab_try['your_guess'] = your_guess
    i_vocab_try['question_time'] = question_time.seconds

    return i_vocab_try


def check_answer(i_vocab_try, all_vocab):

    # Answer analysis
    i_vocab_try['is_it_correct'] = i_vocab_try['your_guess'] == i_vocab_try['word_output_language']

    i_vocab_try['is_it_another_word'] = None
    if not i_vocab_try['is_it_correct']:
        i_vocab_try['is_it_another_word'] = (
                i_vocab_try['your_guess'] in all_vocab[i_vocab_try['output_language']].values
        )

    return i_vocab_try


def if_correct(i_vocab_try):
    """Print statement if correct answer
    """

    print(f"{PrintColors.GREEN}Correct!{PrintColors.ENDC}")

    i_vocab_try['write_it_again'] = None
    i_vocab_try['confused_word'] = None

    return i_vocab_try


def if_not_correct(i_vocab_try, vocab, all_vocab):
    """Print statement + rewrite input if wrong answer

    Parameters
    -----
    i_vocab_try: dict()
        try
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
    vocab: pd.DataFrame
        The vocab table
    all_vocab: pd.DataFrame
        All vocab table

    Return
    -----
    None

    """

    print(
        "{}Sorry{}, it was '{} = {}'".format(
            PrintColors.FAIL, PrintColors.ENDC,
            i_vocab_try['word_input_language'],
            i_vocab_try['word_output_language']
        ))

    i_vocab_try['confused_word'] = None
    if i_vocab_try['is_it_another_word']:

        confused_word = vocab[
            vocab[i_vocab_try['output_language']] == i_vocab_try['your_guess']
        ]

        if len(confused_word) > 0:

            i_vocab_try['confused_word'] = confused_word.iloc[0]['id_vocab']
            confused_word_to_print = confused_word.iloc[0][i_vocab_try['input_language']]

        else:

            i_vocab_try['confused_word'] = 0
            confused_word_all_vocab = all_vocab[
                all_vocab[i_vocab_try['output_language']] == i_vocab_try['your_guess']
                ]
            confused_word_to_print = confused_word_all_vocab.iloc[0][i_vocab_try['input_language']]

        print("{}You confused it{} with '{} = {}'".format(
            PrintColors.WARNING, PrintColors.ENDC,
            i_vocab_try['your_guess'],
            confused_word_to_print
        ))

    write_it_again = input("Write it again   ")
    i_vocab_try['write_it_again'] = write_it_again

    return i_vocab_try


def add_historical_data(historical_data, i_vocab_try):
    """ Add try to historical data

    Parameters
    -----
    historical_data: pd.DataFrame()
        Historical data of the session
    i_vocab_try: pd.DataFrame
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
        'id_vocab': i_vocab_try['id_vocab'],
        'german_word': i_vocab_try['german_word'],
        'english_word': i_vocab_try['english_word'],
        'score_before': i_vocab_try['score_before'],
        'score_before_other_language': i_vocab_try['score_before_other_language'],
        'language_asked': i_vocab_try['output_language'],
        'result': i_vocab_try['is_it_correct'],
        'guess': i_vocab_try['your_guess'],
        'question_time': i_vocab_try['question_time'],
        "write_it_again": i_vocab_try['write_it_again'],
        "is_it_another_word": i_vocab_try['is_it_another_word'],
        "confused_word": i_vocab_try['confused_word'],
        'datetime': i_vocab_try['datetime']
    }, ignore_index=True)

    return historical_data


def update_vocab(vocab, i_vocab_try):
    """ Update the vocab table

    Parameters
    -----
    vocab: pd.DataFrame
        The vocab table
    i_vocab_try: dict()
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

    if i_vocab_try['is_it_correct']:
        vocab.loc[
            vocab["id_vocab"] == i_vocab_try['id_vocab'],
            f"score_{i_vocab_try['input_language']}_{i_vocab_try['output_language']}"
        ] += 1

    if not i_vocab_try['is_it_correct']:
        vocab.loc[
            vocab["id_vocab"] == i_vocab_try['id_vocab'],
            f"score_{i_vocab_try['input_language']}_{i_vocab_try['output_language']}"
        ] -= 1

    vocab.loc[
        vocab["id_vocab"] == i_vocab_try['id_vocab'],
        f"try_session_{i_vocab_try['input_language']}_{i_vocab_try['output_language']}"
    ] = True

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
        historical_data_old = pd.read_csv('data/official/historical_data.csv')

    historical_data_new = pd.concat([historical_data_old, historical_data], axis=0)

    if test:
        historical_data_new.to_csv(f'data/raw/historical_data_{test_name}_after.csv', index=False)
    else:
        historical_data_new.to_csv('data/official/historical_data.csv', index=False)


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

    vocab = vocab.copy()

    del vocab['english_proba']
    del vocab['german_proba']

    if test:
        vocab.to_csv(f'data/raw/german_english_{test_name}_after.csv', index=False)
    else:
        vocab.to_csv('data/official/german_english.csv', index=False)


def info_vocab(test=True, test_name='test'):
    """ Information on the vocab table

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
        probas_next_session = pd.read_csv(f"data/raw/predictions_next_session_{test_name}.csv")
    else:
        vocab = pd.read_csv('data/official/german_english.csv')
        probas_next_session = pd.read_csv('data/official/predictions_next_session.csv')

    vocab = pd.merge(
        vocab,
        probas_next_session,
        on='id_vocab'
    )

    info_vocab_direct(
        vocab,
        model_name=test_name if test else 'official',
        show_plot=True,
        save_plot=True,
        save_folder='data/raw' if test else 'data/official'
    )


def info_vocab_direct(vocab, model_name, show_plot, save_plot, save_folder):

    words_left = vocab[(vocab["english_proba"] < 0.9) | (vocab["german_proba"] < 0.9)]

    print(
        "{}/{} words are considered as known.".format(
            len(vocab) - len(words_left), len(vocab)
        ))

    if len(words_left) < 100:
        print(
            "{}Time to add new words!!{} - only {} words left'".format(
                PrintColors.FAIL, PrintColors.ENDC, len(words_left)
            ))

    plot_comparison_both_probabilities(
        vocab,
        model_name=model_name,
        show_plot=show_plot,
        save_plot=save_plot,
        save_folder=save_folder
    )
