#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as st
from scipy.sparse.linalg import eigs
from scipy.spatial.distance import cdist

import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict
from os.path import basename


class SubspaceAlignedClassifier(object):
    """
    Class of classifiers based on Subspace Alignment.

    Methods contain the alignment itself, classifiers and general utilities.

    Examples
    --------
    | >>>> X = np.random.randn(10, 2)
    | >>>> y = np.vstack((-np.ones((5,)), np.ones((5,))))
    | >>>> Z = np.random.randn(10, 2)
    | >>>> clf = SubspaceAlignedClassifier()
    | >>>> clf.fit(X, y, Z)
    | >>>> preds = clf.predict(Z)

    """

    def __init__(self, loss='logistic', l2=1.0, subspace_dim=1):
        """
        Select a particular type of subspace aligned classifier.

        Parameters
        ----------
        loss : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2 : float
            l2-regularization parameter value (def:0.01)
        sub_space : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        None

        """
        self.loss = loss
        self.l2 = l2
        self.subspace_dim = subspace_dim

        # Initialize untrained classifiers
        if self.loss in ('lr', 'logr', 'logistic'):
            # Logistic regression model
            self.clf = LogisticRegression(C=l2)

        elif self.loss in ('square', 'qd', 'quadratic'):
            # Least-squares model
            self.clf = Ridge(alpha=l2)

        elif self.loss == 'hinge':
            # Linear support vector machine
            self.clf = LinearSVC(C=l2)

        elif self.loss == 'rbfsvc':
            # Radial basis function support vector machine
            self.clf = SVC(C=l2)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented.')

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

    def is_pos_def(self, X):
        """Check for positive definiteness."""
        return np.all(np.linalg.eigvals(X) > 0)

    def subspace_alignment(self, X, Z, subspace_dim=1):
        """
        Compute subspace and alignment matrix.

        Parameters
        ----------
        X : array
            source data set (N samples by D features)
        Z : array
            target data set (M samples by D features)
        subspace_dim : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        V : array
            transformation matrix (D features by D features)
        CX : array
            source principal component coefficients
        CZ : array
            target principal component coefficients

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Check for sufficient samples
        if (N < subspace_dim) or (M < subspace_dim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Compute principal components
        CX = PCA(n_components=subspace_dim).fit(X).components_.T
        CZ = PCA(n_components=subspace_dim).fit(Z).components_.T

        # Aligned source components
        V = np.dot(CX.T, CZ)

        # Return transformation matrix and principal component coefficients
        return V, CX, CZ

    def fit(self, X, Y, Z):
        """
        Fit/train a classifier on data mapped onto transfer components.

        Parameters
        ----------
        X : array
            source data (N samples by D features).
        Y : array
            source labels (N samples by 1).
        Z : array
            target data (M samples by D features).

        Returns
        -------
        None

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Check for sufficient samples
        if (N < self.subspace_dim) or (M < self.subspace_dim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Transfer component analysis
        V, CX, CZ = self.subspace_alignment(X, Z, subspace_dim=self.subspace_dim)

        # Store target subspace
        self.target_subspace = CZ

        # Map source data onto source principal components
        X = np.dot(X, CX)

        # Align source data to target subspace
        X = np.dot(X, V)

        # Train a weighted classifier
        if self.loss in ('lr', 'logr', 'logistic'):
            # Logistic regression model with sample weights
            self.clf.fit(X, Y)

        elif self.loss in ('square', 'qd', 'quadratic'):
            # Least-squares model with sample weights
            self.clf.fit(X, Y)

        elif self.loss == 'hinge':
            # Linear support vector machine with sample weights
            self.clf.fit(X, Y)

        elif self.loss == 'rbfsvc':
            # Radial basis function support vector machine
            self.clf.fit(X, Y)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented')

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

    def predict(self, Z, whiten=False):
        """
        Make predictions on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples by D features)
        whiten : boolean
            whether to whiten new data (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples by 1)

        """
        # Data shape
        M, D = Z.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Check for need to whiten data beforehand
        if whiten:
            Z = st.zscore(Z)

        # Map new target data onto target subspace
        Z = np.dot(Z, self.target_subspace)

        # Call scikit's predict function
        preds = self.clf.predict(Z)

        # For quadratic loss function, correct predictions
        if self.loss == 'quadratic':
            preds = (np.sign(preds)+1)/2.

        # Return predictions array
        return preds

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()


class SemiSubspaceAlignedClassifier(object):
    """
    Class of classifiers based on semi-supervised Subspace Alignment.

    Methods contain the alignment itself, classifiers and general utilities.

    Examples
    --------
    | >>>> X = np.random.randn(10, 2)
    | >>>> y = np.vstack((-np.ones((5,)), np.ones((5,))))
    | >>>> Z = np.random.randn(10, 2)
    | >>>> clf = SubspaceAlignedClassifier()
    | >>>> clf.fit(X, y, Z)
    | >>>> preds = clf.predict(Z)

    """

    def __init__(self, loss='logistic', l2=1.0, subspace_dim=1):
        """
        Select a particular type of subspace aligned classifier.

        Parameters
        ----------
        loss : str
            loss function for weighted classifier, options: 'logistic',
            'quadratic', 'hinge' (def: 'logistic')
        l2 : float
            l2-regularization parameter value (def:0.01)
        sub_space : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        None

        """
        self.loss = loss
        self.l2 = l2
        self.subspace_dim = subspace_dim

        # Initialize untrained classifiers
        if self.loss in ('lr', 'logr', 'logistic'):
            # Logistic regression model
            self.clf = LogisticRegression(C=l2)

        elif self.loss in ('square', 'qd', 'quadratic'):
            # Least-squares model
            self.clf = Ridge(alpha=l2)

        elif self.loss == 'hinge':
            # Linear support vector machine
            self.clf = LinearSVC(C=l2)

        elif self.loss == 'rbfsvc':
            # Radial basis function support vector machine
            self.clf = SVC(C=l2)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented.')

        # Whether model has been trained
        self.is_trained = False

        # Dimensionality of training data
        self.train_data_dim = ''

    def is_pos_def(self, X):
        """Check for positive definiteness."""
        return np.all(np.linalg.eigvals(X) > 0)

    def semi_subspace_alignment(self, X, Y, Z, u, subspace_dim=1):
        """
        Compute subspace and alignment matrix, for each class.

        Parameters
        ----------
        X : array
            source data set (N samples by D features)
        Y : array
            source labels (N samples by 1)
        Z : array
            target data set (M samples by D features)
        u : array
            target labels, first column is index in Z, second column is label
            (m samples by 2)
        subspace_dim : int
            Dimensionality of subspace to retain (def: 1)

        Returns
        -------
        V : array
            transformation matrix (K, D features by D features)
        CX : array
            source principal component coefficients
        CZ : array
            target principal component coefficients

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Check for sufficient samples
        if (N < subspace_dim) or (M < subspace_dim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Number of classes
        K = len(np.unique(Y))

        for k in range(K):

            # Check number of samples per class
            Nk = np.sum(Y == k)

            # Check if subspace dim is too large
            if (Nk < subspace_dim):

                # Reduce subspace dim
                subspace_dim = min(subspace_dim, Nk)

                print('Reducing subspace dim to {}'.format(subspace_dim))

        # Initialize PCA
        pca = PCA(n_components=subspace_dim)

        # Find principal components of target data
        CZ = pca.fit(Z).components_.T

        # Map target data onto principal components
        Z = Z @ CZ

        # Use k-nn to label target samples
        kNN = KNeighborsClassifier(n_neighbors=1)
        U = kNN.fit(Z[u[:, 0], :], u[:, 1]).predict(Z)

        # Preallocate
        CX = []
        V = []

        for k in range(K):

            # Compute source class-conditional principal components
            CX.append(pca.fit(X[Y == k, :]).components_.T)

            # Compute source class-conditional principal components
            CZ_ = pca.fit(Z[U == k, :]).components_.T

            # Aligned source components
            V.append(np.dot(CX[k].T, CZ))

        # Return transformation matrix and principal component coefficients
        return V, CX, CZ

    def fit(self, X, Y, Z, u=None):
        """
        Fit/train a classifier on data mapped onto transfer components.

        Parameters
        ----------
        X : array
            source data (N samples by D features).
        Y : array
            source labels (N samples by 1).
        Z : array
            target data (M samples by D features).
        u : array
            target labels, first column corresponds to index of Z and second
            column corresponds to actual label (number of labels by 2).

        Returns
        -------
        None

        """
        # Data shapes
        N, DX = X.shape
        M, DZ = Z.shape

        # Number of classes
        K = np.unique(Y)

        # Check for sufficient samples
        if (N < self.subspace_dim) or (M < self.subspace_dim):
            raise ValueError('Too few samples for subspace dimensionality.')

        # Assert equivalent dimensionalities
        if not DX == DZ:
            raise ValueError('Dimensionalities of X and Z should be equal.')

        # Transfer component analysis
        V, CX, CZ = self.semi_subspace_alignment(X, Y, Z, u, subspace_dim=self.subspace_dim)

        # Store target subspace
        self.target_subspace = CZ

        for k in range(K):

            # Map source data onto source principal components
            X[Y == k, :] = np.dot(X[Y == k, :], CX[k])

            # Align source data to target subspace
            X[Y == k, :] = np.dot(X[Y == k, :], V[k])

        # Train a weighted classifier
        if self.loss in ('lr', 'logr', 'logistic'):
            # Logistic regression model with sample weights
            self.clf.fit(X, Y)

        elif self.loss in ('square', 'qd', 'quadratic'):
            # Least-squares model with sample weights
            self.clf.fit(X, Y)

        elif self.loss == 'hinge':
            # Linear support vector machine with sample weights
            self.clf.fit(X, Y)

        elif self.loss == 'rbfsvc':
            # Radial basis function support vector machine
            self.clf.fit(X, Y)

        else:
            # Other loss functions are not implemented
            raise NotImplementedError('Loss function not implemented')

        # Mark classifier as trained
        self.is_trained = True

        # Store training data dimensionality
        self.train_data_dim = DX

    def predict(self, Z, whiten=False):
        """
        Make predictions on new dataset.

        Parameters
        ----------
        Z : array
            new data set (M samples by D features)
        whiten : boolean
            whether to whiten new data (def: false)

        Returns
        -------
        preds : array
            label predictions (M samples by 1)

        """
        # Data shape
        M, D = Z.shape

        # If classifier is trained, check for same dimensionality
        if self.is_trained:
            if not self.train_data_dim == D:
                raise ValueError("""Test data is of different dimensionality
                                 than training data.""")

        # Check for need to whiten data beforehand
        if whiten:
            Z = st.zscore(Z)

        # Map new target data onto target subspace
        Z = np.dot(Z, self.target_subspace)

        # Call scikit's predict function
        preds = self.clf.predict(Z)

        # For quadratic loss function, correct predictions
        if self.loss == 'quadratic':
            preds = (np.sign(preds)+1)/2.

        # Return predictions array
        return preds

    def get_params(self):
        """Get classifier parameters."""
        return self.clf.get_params()
