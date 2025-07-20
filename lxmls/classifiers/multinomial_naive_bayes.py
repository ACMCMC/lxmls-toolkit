import numpy as np
import scipy as scipy

import lxmls.classifiers.linear_classifier as lc


class MultinomialNaiveBayes(lc.LinearClassifier):
    def __init__(self, xtype="gaussian"):
        lc.LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = False
        self.smooth_param = 1

    def train(self, x, y):
        # n_docs = no. of documents
        # n_features = no. of unique features
        n_docs, n_features = x.shape

        # classes = a list of possible classes
        classes = np.unique(y)
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]

        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_features, n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
        # prior[0] is the prior probability of a document being of class 0
        # likelihood[4, 0] is the likelihood of the fifth(*) feature being
        # active, given that the document is of class 0
        # (*) recall that Python starts indices at 0, so an index of 4
        # corresponds to the fifth feature!

        # ----------
        # Solution to Exercise 1

        # Naive Bayes:
        # we want to estimate y_hat = argmax(y \in Y) P(Y = y | X = x)
        # which is the same as argmax(y \in Y) P(Y = y, X = x) / P(X = x)
        # and since we're working under the same document, the y that maximizes that is gonna be the same regardless of P(x) because it's always the same
        # so we can just get argmax(y \in Y) P(Y = y, X = x)
        # and that's the same as argmax(y \in Y) P(X = x | Y = y) P(Y = y)

        # Now, the question is, if we have features, how do we define P(X = x)?
        # We can define it as if the probabilities of the tokens in the document were independent (naive Bayes assumption)
        # So P(X = x) becomes \prod_{j = 1}^{J} P(W_j = w_j) where J is the number of features in the document (features) ----- check this

        # prior is P(Y = y)
        # conditional probability is P(X = x | Y = y)

        # Get the priors
        for class_i in range(n_classes):
            docs_in_this_class = (y == class_i).sum()
            prior[class_i] = docs_in_this_class / n_docs

        # Now, get the conditional probabilities given the class of the features
        for class_i in range(n_classes):
            docs_in_this_class = (y == class_i).squeeze()
            features_in_this_class = x[docs_in_this_class]
            features_in_this_class_likelihoods = (
                features_in_this_class.sum(axis=0) / docs_in_this_class.sum()
            )
            likelihood[:, class_i] = features_in_this_class_likelihoods

        # End solution to Exercise 1
        # ----------

        params = np.zeros((n_features + 1, n_classes))
        for i in range(n_classes):
            params[0, i] = np.log(prior[i])
            params[1:, i] = np.nan_to_num(np.log(likelihood[:, i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params


if __name__ == "__main__":
    # This is just a test to see if the code runs without errors
    import lxmls.readers.sentiment_reader as srs

    scr = srs.SentimentCorpus("books")
    mnb = MultinomialNaiveBayes()
    params_nb_sc = mnb.train(scr.train_X, scr.train_y)
    y_pred_train = mnb.test(scr.train_X, params_nb_sc)
    acc_train = mnb.evaluate(scr.train_y, y_pred_train)
    y_pred_test = mnb.test(scr.test_X, params_nb_sc)
    acc_test = mnb.evaluate(scr.test_y, y_pred_test)
    print(
        "Multinomial Naive Bayes Amazon Sentiment Accuracy train: %f test: %f"
        % (acc_train, acc_test)
    )
