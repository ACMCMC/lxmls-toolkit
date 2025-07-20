import numpy as np

import lxmls.classifiers.linear_classifier as lc


class Perceptron(lc.LinearClassifier):
    def __init__(self, nr_epochs=10, learning_rate=1, averaged=True):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.nr_epochs = nr_epochs
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_round = []

    def train(self, x, y, seed=1):
        self.params_per_round = []
        x_orig = x[:, :]
        x = self.add_intercept_term(x)
        nr_x, nr_f = x.shape
        nr_c = np.unique(y).shape[0]
        w = np.zeros((nr_f, nr_c))
        for epoch_nr in range(self.nr_epochs):
            # use seed to generate permutation
            np.random.seed(seed)
            perm = np.random.permutation(nr_x)

            # change the seed so next epoch we don't get the same permutation
            seed += 1

            for nr in perm:
                # Make a prediction with the current model parameters
                this_doc_features = x[nr : nr + 1, :]  # shape: (num_features, )
                class_weighted_features = w.transpose() * this_doc_features
                real_class = y[nr].item()
                chosen_class = class_weighted_features.sum(axis=1).argmax().item()
                # Increase features of the prediction
                w[:, real_class] += this_doc_features.flatten()
                w[:, chosen_class] -= this_doc_features.flatten()

                # If real == chosen, then we'll do nothing (sum and subtract = nothing)

            self.params_per_round.append(w.copy())
            self.trained = True
            y_pred = self.test(x_orig, w)
            acc = self.evaluate(y, y_pred)
            self.trained = False
            print("Rounds: %i Accuracy: %f" % (epoch_nr, acc))
        self.trained = True

        if self.averaged:
            new_w = 0
            for old_w in self.params_per_round:
                new_w += old_w
            new_w /= len(self.params_per_round)
            return new_w
        return w


if __name__ == "__main__":
    import lxmls.readers.simple_data_set as sds

    sd = sds.SimpleDataSet(
        nr_examples=100,
        g1=[[-1, -1], 1],
        g2=[[1, 1], 1],
        balance=0.5,
        split=[0.5, 0, 0.5],
    )

    perc = Perceptron()
    params_perc_sd = perc.train(sd.train_X, sd.train_y)
    y_pred_train = perc.test(sd.train_X, params_perc_sd)
    acc_train = perc.evaluate(sd.train_y, y_pred_train)
    y_pred_test = perc.test(sd.test_X, params_perc_sd)
    acc_test = perc.evaluate(sd.test_y, y_pred_test)
    print(
        "Perceptron Simple Dataset Accuracy train: %f test: %f" % (acc_train, acc_test)
    )
