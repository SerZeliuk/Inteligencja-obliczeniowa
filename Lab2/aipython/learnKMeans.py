# learnKMeans.py - k-means learning
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

from learnProblem import Data_set, Learner, Data_from_file
import random
import matplotlib.pyplot as plt

class K_means_learner(Learner):
    
    def __init__(self,dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        self.random_initialize()
        self.max_display_level = 5 

    def random_initialize(self):
        # class_counts[c] is the number of examples with class=c
        self.class_counts = [0]*self.num_classes
        # feature_sum[f][c] is the sum of the values of feature f for class c
        self.feature_sum = {feat:[0]*self.num_classes
                            for feat in self.dataset.input_features}
        for eg in self.dataset.train:
            cl = random.randrange(self.num_classes) # assign eg to random class
            self.class_counts[cl] += 1
            for feat in self.dataset.input_features:
                self.feature_sum[feat][cl] += feat(eg)
        self.num_iterations = 0
        self.display(1,"Initial class counts: ",self.class_counts)

    def distance(self,cl,eg):
        """distance of the eg from the mean of the class"""
        return sum( (self.class_prediction(feat,cl)-feat(eg))**2
                         for feat in self.dataset.input_features)

    def class_prediction(self,feat,cl):
        """prediction of the class cl on the feature with index feat_ind"""
        if self.class_counts[cl] == 0:
            return 0  # arbitrary prediction
        else:
            return self.feature_sum[feat][cl]/self.class_counts[cl]
        
    def class_of_eg(self,eg):
        """class to which eg is assigned"""
        return (min((self.distance(cl,eg),cl)
                        for cl in range(self.num_classes)))[1]  
               # second element of tuple, which is a class with minimum distance

    def k_means_step(self):
        """Updates the model with one step of k-means. 
        Returns whether the assignment is stable.
        """
        new_class_counts = [0]*self.num_classes
        # feature_sum[f][c] is the sum of the values of feature f for class c
        new_feature_sum = {feat: [0]*self.num_classes
                            for feat in self.dataset.input_features}
        for eg in self.dataset.train:
            cl = self.class_of_eg(eg)
            new_class_counts[cl] += 1
            for feat in self.dataset.input_features:
                new_feature_sum[feat][cl] += feat(eg)
        stable = (new_class_counts == self.class_counts) and (self.feature_sum == new_feature_sum)
        self.class_counts = new_class_counts
        self.feature_sum = new_feature_sum
        self.num_iterations += 1
        return stable
    
        
    def learn(self,n=100):
        """do n steps of k-means, or until convergence"""
        i=0
        stable = False
        while i<n and not stable:
            stable = self.k_means_step()
            i += 1
            self.display(1,"Iteration",self.num_iterations,
                             "class counts: ",self.class_counts," Stable=",stable)
        return stable

    def show_classes(self):
        """sorts the data by the class and prints in order.
        For visualizing small data sets
        """
        class_examples = [[] for i in range(self.num_classes)]
        for eg in self.dataset.train:
            class_examples[self.class_of_eg(eg)].append(eg)
        print("Class","Example",sep='\t')
        for cl in range(self.num_classes):
            for eg in class_examples[cl]:
                print(cl,*eg,sep='\t')
    def plot_error(self, maxstep=20):
        """Plots the sum-of-squares error as a function of the number of steps"""
        plt.ion()
        plt.xlabel("step")
        plt.ylabel("Ave sum-of-squares error")
        train_errors = []
        if self.dataset.test:
            test_errors = []
        for i in range(maxstep):
            train_errors.append( sum(self.distance(self.class_of_eg(eg),eg)
                                         for eg in self.dataset.train)
                                 /len(self.dataset.train))
            if self.dataset.test:
                test_errors.append( sum(self.distance(self.class_of_eg(eg),eg)
                                            for eg in self.dataset.test)
                                     /len(self.dataset.test))
            self.learn(1)
        plt.plot(range(maxstep), train_errors,
                 label=str(self.num_classes)+" classes. Training set")
        if self.dataset.test:
            plt.plot(range(maxstep), test_errors,
                     label=str(self.num_classes)+" classes. Test set")
        plt.legend()
        plt.draw()

# data = Data_from_file('data/emdata1.csv', num_train=10, target_index=2000) # trivial example
data = Data_from_file('data/emdata2.csv', num_train=10, target_index=2000)
# data = Data_from_file('data/emdata0.csv', num_train=14, target_index=2000) # example from textbook
kml = K_means_learner(data,2)
num_iter=4
print("Class assignment after",num_iter,"iterations:")
kml.learn(num_iter); kml.show_classes()

# Plot the error
# km2=K_means_learner(data,2); km2.plot_error(10)   # 2 classes
# km3=K_means_learner(data,3); km3.plot_error(10)   # 3 classes
# km13=K_means_learner(data,10); km13.plot_error(10) # 10 classes

# data = Data_from_file('data/carbool.csv', target_index=2000, one_hot=True)
# kml = K_means_learner(data,3)
# kml.learn(20); kml.show_classes()
# km3=K_means_learner(data,3); km3.plot_error(10)    # 3 classes
# km3=K_means_learner(data,10); km3.plot_error(10)   # 10 classes

