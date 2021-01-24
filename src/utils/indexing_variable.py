

def fit_index(dataset, list_variables):
    """ Mapping between index and category, for categorical variables

    For each (categorical) variable, create 2 dictionaries:
        - index_to_categorical: from the index to the category
        - categorical_to_index: from the category to the index

    Parameters
    ----------
    dataset: pandas.core.frame.DataFrame
        DataFrame with (partly) categorical variables
    list_variables: list(str)
        List of variable names to index

    Returns
    -------
    index: dict
        For each categorical column, we have the 2 mappings: idx2cat & idx2cat
    """

    index = dict()

    for icol in list_variables:

        if icol not in dataset.columns:
            raise RuntimeError(f'{icol} not found in dataframe')

        idx2cat = {ii: jj for ii, jj in enumerate(dataset.loc[:, icol].unique())}
        cat2idx = {jj: ii for ii, jj in idx2cat.items()}

        index[icol] = {
            'index_to_categorical': idx2cat,
            'categorical_to_index': cat2idx
        }

    return index


def map_to_or_from_index(dataset, index, type_conversion):
    """Transform categorical variables to their index

    Parameters
    ----------
    dataset: pandas.core.frame.DataFrame
        DataFrame with categorical variables
    index: dict
        For each categorical column (dict index), we have 2 mappings:
            - index_to_categorical
            - categorical_to_index

    Returns
    -------
    dataset: pandas.core.frame.DataFrame
        Dataframe with the mapping & missing values
    """

    for icol in set(index.keys()).intersection(dataset.columns):

        dataset_init = dataset.copy()

        dataset[icol] = dataset[icol].map(
            lambda x: index[icol][type_conversion].get(x, None)
        )

        missing_index = dataset[icol].isna()

        if sum(missing_index) > 0:

            dataset = dataset[~missing_index]
            print(
                "Missing {} for {} ({} rows): {}".format(
                    type_conversion,
                    icol, sum(missing_index), set(dataset_init[missing_index][icol])
                )
            )

        del dataset_init

    return dataset
