import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaseEnsemble
from spe.misc import *


class ClassifierSPE(BaseEnsemble):
    def __init__(
        self,
        n_estimators=15,
        base_estimator=DecisionTreeClassifier(),
        number_of_bins=15,
        random_seed_value=41,
        estimator_params=tuple(),
    ):
        """


        Parameters
        ----------
        n_estimators : TYPE, optional
            DESCRIPTION. The default is 15.
        base_estimator : TYPE, optional
            DESCRIPTION. The default is DecisionTreeClassifier().
        number_of_bins : TYPE, optional
            DESCRIPTION. The default is 15.
        random_seed_value : TYPE, optional
            DESCRIPTION. The default is 41.
        estimator_params : TYPE, optional
            DESCRIPTION. The default is tuple().

        Returns
        -------
        None.

        """

        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.number_of_bins = number_of_bins
        self.random_seed_value = random_seed_value
        self.estimator_params = estimator_params

    def hardness_function(self, y_true, y_hat):
        """


        Parameters
        ----------
        y_true : TYPE
            DESCRIPTION.
        y_hat : TYPE
            DESCRIPTION.

        Returns
        -------
        result : TYPE
            DESCRIPTION.

        """
        result = np.absolute(y_true - y_hat)
        return result

    def prepare_bins(
        self,
        n_bins,
        bin_space,
        all_hardness,
        major_X,
        minimum_hardness,
        maximum_hardness,
    ):
        """


        Parameters
        ----------
        n_bins : TYPE
            DESCRIPTION.
        bin_space : TYPE
            DESCRIPTION.
        all_hardness : TYPE
            DESCRIPTION.
        major_X : TYPE
            DESCRIPTION.
        minimum_hardness : TYPE
            DESCRIPTION.
        maximum_hardness : TYPE
            DESCRIPTION.

        Returns
        -------
        bins : TYPE
            DESCRIPTION.
        avg_hardness_per_bin : TYPE
            DESCRIPTION.

        """
        bins = []
        avg_hardness_per_bin = []

        for lth_bin in range(n_bins):
            lth_bin_upper = (lth_bin + 1) * bin_space + minimum_hardness
            lth_bin_lower = lth_bin * bin_space + minimum_hardness
            indexes_for_current_bin = (all_hardness < lth_bin_upper) & (
                all_hardness >= lth_bin_lower
            )
            if lth_bin == (n_bins - 1):
                maximum_indexes = get_maximum_indexes(all_hardness, maximum_hardness)
                indexes_for_current_bin = indexes_for_current_bin | maximum_indexes

            bins.append(major_X[indexes_for_current_bin])
            current_mean_bin = all_hardness[indexes_for_current_bin].mean()
            avg_hardness_per_bin.append(current_mean_bin)

        return bins, avg_hardness_per_bin

    def get_alpha(self, current_no, total_no):
        """


        Parameters
        ----------
        current_no : TYPE
            DESCRIPTION.
        total_no : TYPE
            DESCRIPTION.

        Returns
        -------
        alpha : TYPE
            DESCRIPTION.

        """
        alpha = np.tan((1 / 2) * np.pi * (current_no / total_no))
        return alpha

    def get_unnormalized_weights(self, avg_hardness_per_bin, self_paced_factor):
        """


        Parameters
        ----------
        avg_hardness_per_bin : TYPE
            DESCRIPTION.
        self_paced_factor : TYPE
            DESCRIPTION.

        Returns
        -------
        unnormalized_weights : TYPE
            DESCRIPTION.

        """
        unnormalized_weights = 1 / (
            avg_hardness_per_bin + self_paced_factor
        )  # unnormalized sampling weight per bin
        unnormalized_weights = process_array(unnormalized_weights)
        return unnormalized_weights

    def get_undersample_count_per_bin(self, n_min, unnormalized_weights):
        """


        Parameters
        ----------
        n_min : TYPE
            DESCRIPTION.
        unnormalized_weights : TYPE
            DESCRIPTION.

        Returns
        -------
        undersample_count_per_bin : TYPE
            DESCRIPTION.

        """
        percentage = unnormalized_weights / unnormalized_weights.sum()
        undersample_count_per_bin = n_min * percentage
        undersample_count_per_bin = undersample_count_per_bin.astype(int) + 1
        return undersample_count_per_bin

    def undersampling_from_each_bin(self, n_bin, bins, undersample_count_per_bin):
        """


        Parameters
        ----------
        n_bin : TYPE
            DESCRIPTION.
        bins : TYPE
            DESCRIPTION.
        undersample_count_per_bin : TYPE
            DESCRIPTION.

        Returns
        -------
        train_majority_X : TYPE
            DESCRIPTION.
        train_majority_y : TYPE
            DESCRIPTION.

        """

        chosen_samples_per_bin = []

        for lth_bin in range(n_bin):

            current_bin_sample_count = min(
                len(bins[lth_bin]), undersample_count_per_bin[lth_bin]
            )

            if current_bin_sample_count > 0:

                np.random.seed(self.random_seed_value)
                current_bin_limit = len(bins[lth_bin])
                chosen_indexes = np.random.choice(
                    current_bin_limit, current_bin_sample_count, replace=False
                )
                chosen_samples_per_bin.append(bins[lth_bin][chosen_indexes])

        train_majority_X = np.concatenate(chosen_samples_per_bin, axis=0)
        train_majority_y = np.full(train_majority_X.shape[0], 0)

        return train_majority_X, train_majority_y

    def undersampling_with_self_pace(
        self, majority_X, majority_y, minority_X, minority_y, i_estimator
    ):
        """


        Parameters
        ----------
        majority_X : TYPE
            DESCRIPTION.
        majority_y : TYPE
            DESCRIPTION.
        minority_X : TYPE
            DESCRIPTION.
        minority_y : TYPE
            DESCRIPTION.
        i_estimator : TYPE
            DESCRIPTION.

        Returns
        -------
        train_X : TYPE
            DESCRIPTION.
        train_y : TYPE
            DESCRIPTION.

        """

        all_hardness = self.hardness_function(
            majority_y, self.majority_y_prediction[:, self.minor_label]
        )  # calculate the hardness of each majority sample
        minimum_hardness = all_hardness.min()
        maximum_hardness = all_hardness.max()

        if maximum_hardness > minimum_hardness:  # dividing majority samples into k bins
            hardness_gap = maximum_hardness - minimum_hardness

            space_per_bin = (hardness_gap) / self.number_of_bins

            """
            Cut majority set into k bins and get average hardness contirubtion per bin
            """
            bins, avg_hardness_per_bin = self.prepare_bins(
                self.number_of_bins,
                space_per_bin,
                all_hardness,
                majority_X,
                minimum_hardness,
                maximum_hardness,
            )

            """
            Update self-paed factor alpha
            """
            current_estimator_no = i_estimator
            n_estimators = self.n_estimators - 1

            self_paced_factor = self.get_alpha(
                current_estimator_no, n_estimators
            )  # getting alpha for current estimator

            """
            Unnormalized sampling weight per bin
            """
            unnormalized_weights = self.get_unnormalized_weights(
                avg_hardness_per_bin, self_paced_factor
            )

            """
            Under sample count per bin
            """
            undersample_count_per_bin = self.get_undersample_count_per_bin(
                len(minority_X), unnormalized_weights
            )

            """
            Under sampling from each bin
            """
            train_majority_X, train_majority_y = self.undersampling_from_each_bin(
                self.number_of_bins, bins, undersample_count_per_bin
            )

            """
            Undersampled new train X and y
            """
            train_X, train_y = stack_majority_minority(
                train_majority_X, minority_X, train_majority_y, minority_y
            )
        else:
            train_X, train_y = random_sampling(
                majority_X, majority_y, minority_X, minority_y, self.random_seed_value
            )  # if all the hardness are same then do rnadom sampling

        return train_X, train_y

    def get_new_majority_prediction(self, majority_y_prediction, majority_X):
        """


        Parameters
        ----------
        majority_y_prediction : TYPE
            DESCRIPTION.
        majority_X : TYPE
            DESCRIPTION.

        Returns
        -------
        majority_y_prediction : TYPE
            DESCRIPTION.

        """

        start = self.n_buffered_estimators_
        end = len(self.estimators_)

        for index in range(start, end):
            current_majority_prediction = self.estimators_[index].predict_proba(
                majority_X
            )
            majority_y_prediction = (
                majority_y_prediction * index + current_majority_prediction
            )
            majority_y_prediction = majority_y_prediction / (index + 1)

        return majority_y_prediction

    def latest_majority_predictions(self, majority_X):
        """


        Parameters
        ----------
        majority_X : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if self.n_buffered_estimators_ == 0:
            self.majority_y_prediction = get_filled_matrix(
                self._n_samples_maj, self.n_classes_, 1 / 2
            )

        majority_y_prediction = self.majority_y_prediction

        self.majority_y_prediction = self.get_new_majority_prediction(
            majority_y_prediction, majority_X
        )

        self.n_buffered_estimators_ = len(self.estimators_)

        return

    def data_initialization(self, X, y):
        """


        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.major_label = 0
        self.minor_label = 1
        self.n_buffered_estimators_ = 0
        self.features_ = np.arange(X.shape[1])
        class_list, y = np.unique(y, return_inverse=True)
        self.classes_ = class_list
        self.n_classes_ = len(class_list)

        return

    def get_max_min(self, X, y):
        """


        Parameters
        ----------
        X : TYPE
            DESCRIPTION.
        y : TYPE
            DESCRIPTION.

        Returns
        -------
        major_X : TYPE
            DESCRIPTION.
        major_y : TYPE
            DESCRIPTION.
        minor_X : TYPE
            DESCRIPTION.
        minor_y : TYPE
            DESCRIPTION.

        """

        major_index = y == self.major_label
        minor_index = y == self.minor_label

        self._n_samples_maj = major_index.sum()
        self._n_samples_min = minor_index.sum()

        major_X = X[major_index]
        major_y = y[major_index]
        minor_X = X[minor_index]
        minor_y = y[minor_index]

        return major_X, major_y, minor_X, minor_y

    def fit(self, train_X, train_y):
        """


        Parameters
        ----------
        train_X : TYPE
            DESCRIPTION.
        train_y : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        self._validate_estimator()

        self.data_initialization(train_X, train_y)

        self.major_X, self.major_y, self.minor_X, self.minor_y = self.get_max_min(
            train_X, train_y
        )

        self.estimators_ = []
        self.estimators_features_ = []

        # Loop start
        iterations = range(self.n_estimators)
        # line 3 the for loop:
        for i_iter in iterations:

            # line 4
            """
            Set majority predictions
            """
            self.latest_majority_predictions(self.major_X)
            """
            Self paced undersamples with hardness function
            """
            train_X, train_y = self.undersampling_with_self_pace(
                self.major_X, self.major_y, self.minor_X, self.minor_y, i_iter
            )
            """
            Prepare current estimator and add this estimator to estimator list
            """
            estimator = self._make_estimator(
                append=True, random_state=self.random_seed_value
            )
            """
            Fitting model
            """
            estimator.fit(train_X, train_y)
            """
            Saving feature set for current estimator
            """
            self.estimators_features_.append(self.features_)

        return self

    def predict_proba(self, test_X):
        """


        Parameters
        ----------
        test_X : TYPE
            DESCRIPTION.

        Returns
        -------
        all_probabilities : TYPE
            DESCRIPTION.

        """

        all_estimators, all_estimators_features, n_classes = (
            self.estimators_,
            self.estimators_features_,
            self.n_classes_,
        )

        all_probabilities = get_filled_matrix(test_X.shape[0], n_classes, 0.0)

        for current_estimator, current_features in zip(
            all_estimators, all_estimators_features
        ):

            input_X = test_X[:, current_features]

            if hasattr(current_estimator, "predict_proba"):
                current_estimators_probability = current_estimator.predict_proba(
                    input_X
                )
                all_probabilities += current_estimators_probability

            else:
                current_estimators_prediction = current_estimator.predict(input_X)

                for sample_index in range(test_X.shape[0]):
                    all_probabilities[
                        sample_index, current_estimators_prediction[sample_index]
                    ] += 1

        all_probabilities = all_probabilities / self.n_estimators

        return all_probabilities

    def predict(self, test_X):
        """


        Parameters
        ----------
        test_X : TYPE
            DESCRIPTION.

        Returns
        -------
        predicted_class : TYPE
            DESCRIPTION.

        """

        y_prediction = self.predict_proba(test_X)
        get_max_prob = np.argmax(y_prediction, axis=1)
        predicted_class = self.classes_.take(get_max_prob, axis=0)

        return predicted_class
