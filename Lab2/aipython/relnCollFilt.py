# relnCollFilt.py - Latent Property-based Collaborative Filtering
# AIFCA Python code Version 0.9.15 Documentation at https://aipython.org
# Download the zip file and read aipython.pdf for documentation

# Artificial Intelligence: Foundations of Computational Agents https://artint.info
# Copyright 2017-2024 David L. Poole and Alan K. Mackworth
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# See: https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en

import random
import matplotlib.pyplot as plt
import urllib.request
from learnProblem import Learner
from display import Displayable

class Rating_set(Displayable):
    """A rating contains:
    training_data: list of (user, item, rating) triples
    test_data: list of (user, item, rating) triples
    """
    def __init__(self, training_data, test_data):
        self.training_data = training_data
        self.test_data = test_data

grades_rs = Rating_set( # 3='A', 2='B', 1='C'
    [('s1','c1',3),    # training data
     ('s2','c1',1),
     ('s1','c2',2),
     ('s2','c3',2),
     ('s3','c2',2),
     ('s4','c3',2)],
    [('s3','c4',3),    # test data
      ('s4','c4',1)])

class CF_learner(Learner):
    def __init__(self,
                 rating_set,            # a Rating_set 
                 step_size = 0.01,      # gradient descent step size
                 regularization = 1.0,  # L2 regularization for full dataset
                 num_properties = 10,   # number of hidden properties
                 property_range = 0.02  # properties are initialized to be between
                                        # -property_range and property_range
                 ):
        self.rating_set = rating_set
        self.training_data = rating_set.training_data
        self.test_data = self.rating_set.test_data
        self.step_size = step_size
        self.regularization = regularization 
        self.num_properties = num_properties
        self.num_ratings = len(self.training_data)
        self.ave_rating = (sum(r for (u,i,r) in self.training_data)
                           /self.num_ratings)
        self.users = {u for (u,i,r) in self.training_data}
        self.items = {i for (u,i,r) in self.training_data}
        self.user_bias = {u:0 for u in self.users}
        self.item_bias = {i:0 for i in self.items}
        self.user_prop = {u:[random.uniform(-property_range,property_range)
                              for p in range(num_properties)]
                             for u in self.users}
        self.item_prop = {i:[random.uniform(-property_range,property_range)
                              for p in range(num_properties)]
                             for i in self.items}
        # the _delta variables are the changes internal to a batch:
        self.user_bias_delta = {u:0 for u in self.users}
        self.item_bias_delta = {i:0 for i in self.items}
        self.user_prop_delta = {u:[0 for p in range(num_properties)]
                                    for u in self.users}
        self.item_prop_delta = {i:[0 for p in range(num_properties)]
                                    for i in self.items}
        # zeros is used for users and items not in the training set
        self.zeros = [0 for p in range(num_properties)]
        self.epoch = 0
        self.display(1, "Predict mean:" "(Ave Abs,AveSumSq)",
                    "training =",self.eval2string(self.training_data, useMean=True),
                    "test =",self.eval2string(self.test_data, useMean=True))

    def prediction(self,user,item):
        """Returns prediction for this user on this item.
        The use of .get() is to handle users or items in test set but not in the training set.
        """
        if user in self.user_bias: # user in training set
            if item in self.item_bias: # item in training set
                return (self.ave_rating
                       + self.user_bias[user]
                       + self.item_bias[item]
                       + sum([self.user_prop[user][p]*self.item_prop[item][p]
                       for p in range(self.num_properties)]))
            else:  # training set contains user but not item
                return (self.ave_rating + self.user_bias[user])
        elif item in self.item_bias: # training set contains item but not user
            return self.ave_rating + self.item_bias[item]
        else:
            return self.ave_rating

    def learn(self, num_epochs = 50, batch_size=1000):    
        """ do (approximately) num_epochs iterations through the dataset
        batch_size is the size of each batch of stochastic gradient gradient descent.
        """
        batch_size = min(batch_size, len(self.training_data))
        batch_per_epoch = len(self.training_data) // batch_size # approximate
        num_iter = batch_per_epoch*num_epochs
        reglz = self.step_size*self.regularization*batch_size/len(self.training_data) #regularization per batch
        
        for i in range(num_iter):
            if i % batch_per_epoch == 0:
                self.epoch += 1
                self.display(1,"Epoch", self.epoch, "(Ave Abs,AveSumSq)",
                            "training =",self.eval2string(self.training_data),
                            "test =",self.eval2string(self.test_data))
            # determine errors for a batch
            for (user,item,rating) in random.sample(self.training_data, batch_size):
                error = self.prediction(user,item) - rating
                self.user_bias_delta[user] += error
                self.item_bias_delta[item] += error
                for p in range(self.num_properties):
                    self.user_prop_delta[user][p] += error*self.item_prop[item][p]
                    self.item_prop_delta[item][p] += error*self.user_prop[user][p]
            # Update all parameters
            for user in self.users:
                self.user_bias[user] -= (self.step_size*self.user_bias_delta[user]
                                         +reglz*self.user_bias[user])
                self.user_bias_delta[user] = 0
                for p in range(self.num_properties):
                    self.user_prop[user][p] -= (self.step_size*self.user_prop_delta[user][p]
                                                + reglz*self.user_prop[user][p])
                    self.user_prop_delta[user][p] = 0
            for item in self.items:
                self.item_bias[item] -= (self.step_size*self.item_bias_delta[item]
                                        + reglz*self.item_bias[item])
                self.item_bias_delta[item] = 0
                for p in range(self.num_properties):
                    self.item_prop[item][p] -= (self.step_size*self.item_prop_delta[item][p]
                                              + reglz*self.item_prop[item][p])
                    self.item_prop_delta[item][p] = 0

    def evaluate(self, ratings, useMean=False):
        """returns (average_absolute_error, average_sum_squares_error) for ratings
        """
        abs_error = 0
        sumsq_error = 0
        if not ratings: return (0,0)
        for (user,item,rating) in ratings:
            prediction = self.ave_rating if useMean else self.prediction(user,item)
            error = prediction - rating
            abs_error += abs(error)
            sumsq_error += error * error
        return abs_error/len(ratings), sumsq_error/len(ratings)

    def eval2string(self, *args, **nargs):
        """returns a string form of evaluate, with fewer digits
        """
        (abs,ssq) = self.evaluate(*args, **nargs)
        return f"({abs:.4f}, {ssq:.4f})"
        
#lg = CF_learner(grades_rs,step_size = 0.1, regularization = 0.01,  num_properties = 1)
#lg.learn(num_epochs = 500)
# lg.item_bias
# lg.user_bias
# lg.plot_property(0,plot_all=True) # can you explain why?

    def plot_predictions(self, examples="test"):
        """
        examples is either "test" or "training" or the actual examples
        """
        if examples == "test":
            examples = self.test_data
        elif examples == "training":
            examples = self.training_data
        plt.ion()
        plt.xlabel("prediction")
        plt.ylabel("cumulative proportion")
        self.actuals = [[] for r in range(0,6)]
        for (user,item,rating) in examples:
            self.actuals[rating].append(self.prediction(user,item))
        for rating in range(1,6):
            self.actuals[rating].sort()
            numrat=len(self.actuals[rating])
            yvals = [i/numrat for i in range(numrat)]
            plt.plot(self.actuals[rating], yvals, label="rating="+str(rating))
        plt.legend()
        plt.draw()
        
    def plot_property(self,
                     p,               # property
                     plot_all=False,  # true if all points should be plotted
                     num_points=200   # number of random points plotted if not all
                     ):
        """plot some of the user-movie ratings,
        if plot_all is true
        num_points is the number of points selected at random plotted.

        the plot has the users on the x-axis sorted by their value on property p and
        with the items on the y-axis sorted by their value on property p and 
        the ratings plotted at the corresponding x-y position.
        """
        plt.ion()
        plt.xlabel("users")
        plt.ylabel("items")
        user_vals = [self.user_prop[u][p]
                     for u in self.users]
        item_vals = [self.item_prop[i][p]
                     for i in self.items]
        plt.axis([min(user_vals)-0.02,
                  max(user_vals)+0.05,
                  min(item_vals)-0.02,
                  max(item_vals)+0.05])
        if plot_all:
            for (u,i,r) in self.training_data:
                plt.text(self.user_prop[u][p],
                         self.item_prop[i][p],
                         str(r))
        else:
            for i in range(num_points):
                (u,i,r) = random.choice(self.training_data)
                plt.text(self.user_prop[u][p],
                         self.item_prop[i][p],
                         str(r))
        plt.show()

class Rating_set_from_file(Rating_set):
    def __init__(self,
                 date_split=892000000,
                 local_file=False,
                 url="http://files.grouplens.org/datasets/movielens/ml-100k/u.data",
                 file_name="u.data"):
        self.display(1,"Collaborative Filtering Dataset.  Reading...")
        if local_file:
            lines = open(file_name,'r')
        else:
            lines = (line.decode('utf-8') for line in urllib.request.urlopen(url))
        all_ratings = (tuple(int(e) for e in line.strip().split('\t'))
                        for line in lines)
        self.training_data = []
        self.training_stats = {1:0, 2:0, 3:0, 4:0 ,5:0}
        self.test_data = []
        self.test_stats = {1:0, 2:0, 3:0, 4:0 ,5:0}
        for (user,item,rating,timestamp) in all_ratings:
            if timestamp < date_split:   # rate[3] is timestamp
                self.training_data.append((user,item,rating))
                self.training_stats[rating] += 1
            else:
                self.test_data.append((user,item,rating))
                self.test_stats[rating] += 1
        self.display(1,"...read:", len(self.training_data),"training ratings and",
                len(self.test_data),"test ratings")
        tr_users = {user for (user,item,rating) in self.training_data}
        test_users = {user for (user,item,rating) in self.test_data}
        self.display(1,"users:",len(tr_users),"training,",len(test_users),"test,",
                     len(tr_users & test_users),"in common")
        tr_items = {item for (user,item,rating) in self.training_data}
        test_items = {item for (user,item,rating) in self.test_data}
        self.display(1,"items:",len(tr_items),"training,",len(test_items),"test,",
                     len(tr_items & test_items),"in common")
        self.display(1,"Rating statistics for training set: ",self.training_stats)
        self.display(1,"Rating statistics for test set: ",self.test_stats)

class Rating_set_top_subset(Rating_set):
    
    def __init__(self, rating_set, num_items = (20,40), num_users = (20,24)):
        """Returns a subset of the ratings by picking the most rated items,
        and then the users that have most ratings on these, and then all of the
        ratings that involve these users and items.
        num_items is (ni,si) which selects ni users at random from the top si users
        num_users is (nu,su) which selects nu items at random from the top su items
        """
        (ni, si) = num_items
        (nu, su) = num_users
        items = {item for (user,item,rating) in rating_set.training_data}
        item_counts = {i:0 for i in items}
        for (user,item,rating) in rating_set.training_data:
            item_counts[item] += 1

        items_sorted = sorted((item_counts[i],i) for i in items)
        top_items = random.sample([item for (count, item) in items_sorted[-si:]], ni)
        set_top_items = set(top_items)

        users = {user for (user,item,rating) in rating_set.training_data}
        user_counts = {u:0 for u in users}
        for (user,item,rating) in rating_set.training_data:
            if item in set_top_items:
                user_counts[user] += 1

        users_sorted = sorted((user_counts[u],u) for u in users)
        top_users = random.sample([user for (count, user) in users_sorted[-su:]], nu)
        set_top_users = set(top_users)
        
        self.training_data = [ (user,item,rating)
                         for (user,item,rating) in rating_set.training_data
                         if user in set_top_users and item in set_top_items]
        self.test_data = []

movielens = Rating_set_from_file()
learner1 = CF_learner(movielens, num_properties = 1)
# learner10 = CF_learner(movielens, num_properties = 10)
# learner1.learn(50)
# learner1.plot_predictions(examples = "training")
# learner1.plot_predictions(examples = "test")
# learner1.plot_property(0)
# movielens_subset = Rating_set_top_subset(movielens,num_items = (20,40), num_users = (20,40))
# learner_s = CF_learner(movielens_subset, num_properties=1)
# learner_s.learn(1000)
# learner_s.plot_property(0,plot_all=True)

