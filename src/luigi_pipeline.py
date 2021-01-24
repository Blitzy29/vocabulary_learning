import datetime
import dill
import luigi
import pandas as pd
import os
from shutil import copyfile

from src.data.collect_data import info_vocab_direct
import src.data.get_dataset as get_dataset
import src.data.make_dataset as make_dataset
from src.data.make_historical_features import create_historical_features
from src.data.make_vocab_features import create_vocab_features

from src.models.gradient_boosting import ModelGradientBoosting

from src.data.make_predictions_next_session import (
    make_and_save_predictions_next_session,
)

today = datetime.datetime.today().strftime("%Y%m%d")


# Task 1: create a new folder where the pipeline will work

class CreatePipelineFolder(luigi.Task):
    name = luigi.Parameter(default="create pipeline folder")
    folder_name = luigi.Parameter(default="test_")

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/01_create_pipeline_folder.txt".format(self.folder_name, today)
        )

    def run(self):
        newpath = r"data/pipeline/{}_{}".format(self.folder_name, today)
        if not os.path.exists(newpath):
            os.makedirs(newpath)

        with self.output().open("w") as f:
            f.write("Pipeline folder created.")


# Task 2: copy the historical data and vocab data from `official` to pipeline folder

class CopyFilesIntoPipeline(luigi.Task):
    name = luigi.Parameter(default="copy files into pipeline")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return CreatePipelineFolder(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/02_copy_files_into_pipeline.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        for i_file in ["german_english.csv", "historical_data.csv"]:
            copyfile(
                src=r"data/official/{}".format(i_file),
                dst=r"data/pipeline/{}_{}/{}".format(self.folder_name, today, i_file),
            )

        with self.output().open("w") as f:
            f.write("Files into pipeline copied.")


# Task 3A: create historical dataset features

class CreateHistoricalDatasetFeatures(luigi.Task):
    name = luigi.Parameter(default="create historical dataset features.")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return CopyFilesIntoPipeline(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/03A_create_historical_dataset_features.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        historical_data = get_dataset.get_historical_data(
            historical_data_path="data/pipeline/{}_{}/{}".format(
                self.folder_name, today, "historical_data.csv"
            ),
        )

        historical_data = create_historical_features(historical_data)

        historical_data_features_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "historical_dataset_features.pkl"
        )
        with open(historical_data_features_path, "wb") as file:
            dill.dump(historical_data, file)

        with self.output().open("w") as f:
            f.write("Historical dataset features created.")


# Task 3B: create vocab dataset features

class CreateVocabDatasetFeatures(luigi.Task):
    name = luigi.Parameter(default="create vocab dataset features.")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return CopyFilesIntoPipeline(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/03B_create_vocab_dataset_features.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        vocab = get_dataset.get_vocab(
            vocab_path="data/pipeline/{}_{}/{}".format(
                self.folder_name, today, "german_english.csv"
            ),
            list_columns="all",
        )
        vocab = create_vocab_features(vocab)

        vocab_data_features_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "vocab_dataset_features.pkl"
        )
        with open(vocab_data_features_path, "wb") as file:
            dill.dump(vocab, file)

        with self.output().open("w") as f:
            f.write("Vocab dataset features created.")


# Task 4: merge features dataset together

class MergeFeaturesTogether(luigi.Task):
    name = luigi.Parameter(
        default="merge historical and vocab dataset features together."
    )
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return (
            CreateHistoricalDatasetFeatures(folder_name=self.folder_name),
            CreateVocabDatasetFeatures(folder_name=self.folder_name),
        )

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/04_merge_features_together.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        historical_data_features_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "historical_dataset_features.pkl"
        )
        with open(historical_data_features_path, "rb") as input_file:
            historical_data_features = dill.load(input_file)

        vocab_data_features_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "vocab_dataset_features.pkl"
        )
        with open(vocab_data_features_path, "rb") as input_file:
            vocab_data_features = dill.load(input_file)

        dataset = make_dataset.merge_feature_datasets(
            historical_data_features, vocab_data_features
        )

        vardict = make_dataset.get_vardict()
        dataset = make_dataset.transform_type(dataset, vardict)

        dataset_path = "data/pipeline/{}_{}/{}".format(self.folder_name, today, "dataset.pkl")

        with open(dataset_path, "wb") as file:
            dill.dump(dataset, file)

        with self.output().open("w") as f:
            f.write("Historical and Vocab dataset features merged together.")


# Task 5: split into train/validation/test dataset

class SplitDatasetIntoTrainValidTest(luigi.Task):
    name = luigi.Parameter(default="split the dataset into train/valid/test datasets.")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return MergeFeaturesTogether(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/05_split_dataset_into_train_valid_test.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        dataset_path = "data/pipeline/{}_{}/{}".format(self.folder_name, today, "dataset.pkl")
        with open(dataset_path, "rb") as input_file:
            dataset = dill.load(input_file)

        train_dataset_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "train_dataset.pkl"
        )
        valid_dataset_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "valid_dataset.pkl"
        )
        test_dataset_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "test_dataset.pkl"
        )

        make_dataset.split_train_valid_test_dataset(
            dataset, train_dataset_path, valid_dataset_path, test_dataset_path
        )

        with self.output().open("w") as f:
            f.write("Train, validation and test datasets splitted.")


# Task 6: Train model

class TrainGradientBoostingModel(luigi.Task):
    name = luigi.Parameter(default="train gradient boosting model")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return SplitDatasetIntoTrainValidTest(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/06_train_gradient_boosting_model.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        path_dataset_train = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "train_dataset.pkl"
        )
        with open(path_dataset_train, "rb") as input_file:
            dataset_train = dill.load(input_file)

        model = ModelGradientBoosting()
        dataset_train = model.preprocessing_training(dataset_train)
        model.train(dataset_train)

        path_dataset_valid = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "valid_dataset.pkl"
        )
        with open(path_dataset_valid, "rb") as input_file:
            dataset_valid = dill.load(input_file)

        model.predict_and_show_results(
            dataset_valid, save_folder="data/pipeline/{}_{}".format(self.folder_name, today)
        )

        path_model = "data/pipeline/{}_{}/{}".format(self.folder_name, today, "model.pkl")
        with open(path_model, "wb") as file:
            dill.dump(model, file)

        with self.output().open("w") as f:
            f.write("Model trained.")


# Task 7: create next session historical and vocab features

class CreateNewSessionFeaturesDataset(luigi.Task):
    name = luigi.Parameter(default="create new session dataset features.")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return CopyFilesIntoPipeline(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/07_create_new_session_dataset_features.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        historical_data = get_dataset.get_historical_data(
            historical_data_path="data/pipeline/{}_{}/{}".format(
                self.folder_name, today, "historical_data.csv"
            ),
        )

        vocab = get_dataset.get_vocab(
            vocab_path="data/pipeline/{}_{}/{}".format(
                self.folder_name, today, "german_english.csv"
            ),
            list_columns="all",
        )

        dataset_predictions_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "new_session_features_dataset.pkl"
        )
        make_dataset.create_dataset_new_session(
            dataset_predictions_path,
            historical_data=historical_data,
            vocab_to_predict=vocab,
        )

        with self.output().open("w") as f:
            f.write("New session features dataset created.")


# Task 8: Make predictions

class MakePredictions(luigi.Task):
    name = luigi.Parameter(default="make predictions")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return (
            TrainGradientBoostingModel(folder_name=self.folder_name),
            CreateNewSessionFeaturesDataset(folder_name=self.folder_name),
        )

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/08_make_predictions.txt".format(self.folder_name, today)
        )

    def run(self):
        path_model = "data/pipeline/{}_{}/{}".format(self.folder_name, today, "model.pkl")
        with open(path_model, "rb") as input_file:
            model = dill.load(input_file)

        next_session_features_dataset_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "new_session_features_dataset.pkl"
        )
        with open(next_session_features_dataset_path, "rb") as input_file:
            next_session_features_dataset = dill.load(input_file)

        next_session_probas_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "next_session_probas_path.pkl"
        )

        make_and_save_predictions_next_session(
            model=model,
            next_session_features_dataset=next_session_features_dataset,
            probas_next_session_path=next_session_probas_path,
        )

        with self.output().open("w") as f:
            f.write("Predictions made.")


# Task 9: copy predictions to csv and official

class CopyPredictionsToOfficial(luigi.Task):
    name = luigi.Parameter(default="copy predictions to 'official' as a csv")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return MakePredictions(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/09_copy_predictions_to_official.txt".format(
                self.folder_name, today
            )
        )

    def run(self):
        next_session_probas_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "next_session_probas_path.pkl"
        )
        with open(next_session_probas_path, "rb") as input_file:
            next_session_probas = dill.load(input_file)

        predictions_next_session_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "predictions_next_session.csv"
        )
        next_session_probas.to_csv(predictions_next_session_path, index=False)

        copyfile(
            src=r"data/pipeline/{}_{}/{}".format(
                self.folder_name, today, "predictions_next_session.csv"
            ),
            dst=r"data/official/{}".format("predictions_next_session.csv"),
        )

        with self.output().open("w") as f:
            f.write("Predictions copied in csv format to official.")


# Task 10: PrepareNextSession

class PrepareNextSession(luigi.Task):
    folder_name = luigi.Parameter(default="test_")

    name = luigi.Parameter(
        default="merge historical and vocab dataset features together."
    )

    def requires(self):
        return CopyPredictionsToOfficial(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/10_prepare_next_session.txt".format(self.folder_name, today)
        )

    def run(self):
        with self.output().open("w") as f:
            f.write("Next session ready.")


# Task 10bis: Info on vocab post review

class InformationVocabPostReview(luigi.Task):
    name = luigi.Parameter(default="information post review on vocabulary and predictions")
    folder_name = luigi.Parameter(default="test_")

    def requires(self):
        return MakePredictions(folder_name=self.folder_name)

    def output(self):
        return luigi.LocalTarget(
            "data/pipeline/{}_{}/10bis_info_vocab_pot_review.txt".format(self.folder_name, today)
        )

    def run(self):
        vocab = get_dataset.get_vocab(
            vocab_path="data/pipeline/{}_{}/{}".format(
                self.folder_name, today, "german_english.csv"
            ),
            list_columns="all",
        )

        next_session_probas_path = "data/pipeline/{}_{}/{}".format(
            self.folder_name, today, "next_session_probas_path.pkl"
        )
        with open(next_session_probas_path, "rb") as input_file:
            probas_next_session = dill.load(input_file)

        vocab = pd.merge(
            vocab,
            probas_next_session,
            on='id_vocab'
        )

        info_vocab_direct(
            vocab=vocab,
            model_name=self.folder_name,
            show_plot=False,
            save_plot=True,
            save_folder='data/pipeline/{}_{}'.format(self.folder_name, today)
        )

        with self.output().open("w") as f:
            f.write("Next session ready.")
