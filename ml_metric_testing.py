import logging
import ConfigParser
import csv
import functools
import getpass
import koji
import datetime
import time
from requests.exceptions import ConnectionError
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

class Brew():

    def __init__(self):
	self.client = koji.ClientSession('http://brewhub.engineering.redhat.com/brewhub')
    
    def retry(timeout=240, interval=60, wait_on=Exception):
        """ A decorator that allows to retry a section of code...
        ...until success or timeout.
        """
        def wrapper(function):
            @functools.wraps(function)
            def inner(*args, **kwargs):
                start = time.time()
                while True:
                    if (time.time() - start) >= timeout:
                        raise  # This re-raises the last exception.
                    try:
                        return function(*args, **kwargs)
                    except wait_on as e:
                        print("Exception %r raised from %r.  Retry in %rs" % (
                            e, function, interval))
                        time.sleep(interval)
            return inner
        return wrapper

    def iterator(self, start_date, end_date):
        current_date = start_date
        delta = datetime.timedelta(days=1)
        while current_date < end_date:
            yield current_date
            current_date += delta

    @retry(wait_on=ConnectionError)
    def find_data(self):
        # fetch builds from Teiid
	docs = []
	start_date = datetime.datetime.strptime(
            '2013-07-30', '%Y-%m-%d').date()
        end_date = datetime.date.today()
	previous_date = start_date
	count = 0
        for date in self.iterator(start_date, end_date):
            builds = self.client.listBuilds(
                    completeAfter=str(previous_date), completeBefore=str(date))
            docs.extend(builds)
            previous_date = date
	    count += 1
	    if count%5 == 0:
		time.sleep(1)
	return docs

    def copy_data_in_csv(self, docs):

        # This is just faster in terms of retrieval. The calculation of diff and removal of None 
        # would be easier in dataframes.
        the_file = open("metric.csv", "w")
        writer = csv.DictWriter(the_file, docs[0].keys())
        writer.writeheader()
        writer.writerows(docs)
        data_df = pd.read_csv("metric.csv")
	data_df['waiting_time'] = data_df.creation_ts - data_df.start_ts
	self.linear_regression(data_df)
	the_file.close()

    def linear_regression(self, data_df):

	# Pre-process the data to bring it in the suitable format
	regression_df = data_df[['extra', 'package_id', 'build_id', 'owner_id', 'creation_event_id', 'state','start_ts','creation_ts']]
	regression_df['extra'] = regression_df['extra'].replace("{", 1, regex=True)
	regression_df['extra'] = regression_df['extra'].fillna(0)
	regression_df = regression_df.dropna()
	wait_time_df = data_df[['waiting_time']]
	wait_time_df = wait_time_df.dropna()
	regression_df['start_ts'] = (regression_df['start_ts'] - regression_df['start_ts'].mean())/regression_df['start_ts'].std(ddof=0)
	regression_df['creation_ts'] = (regression_df['creation_ts'] - regression_df['creation_ts'].mean())/regression_df['creation_ts'].std(ddof=0)
	print ("regression_df", regression_df)
	print ("wait_time_df", wait_time_df)

	# Split the data into testing and training set
	X_train, X_test, y_train, y_test = train_test_split(regression_df, wait_time_df, test_size=0.33, random_state=42)

	X_train.reset_index(inplace=True)
	X_test.reset_index(inplace=True)
	y_train.reset_index(inplace=True)
	y_test.reset_index(inplace=True)

	
	np.random.seed(0)

	classifiers = dict(ols=linear_model.LinearRegression(),
                   ridge=linear_model.Ridge(alpha=.1))

	fignum = 1
	for name, clf in classifiers.items():
	    fig = plt.figure(fignum, figsize=(4, 3))
	    plt.clf()
	    plt.title(name)
	    ax = plt.axes([.12, .12, .8, .8])

        for _ in range(6):
            this_X = .1 * np.random.normal(size=(2, 1)) + X_train
            clf.fit(this_X, y_train)

        ax.plot(X_test, clf.predict(X_test), color='.5')
        ax.scatter(this_X, y_train, s=3, c='.5', marker='o', zorder=10)

    	clf.fit(X_train, y_train)
    	ax.plot(X_test, clf.predict(X_test), linewidth=2, color='blue')
    	ax.scatter(X_train, y_train, s=30, c='r', marker='+', zorder=10)

    	ax.set_xticks(())
    	ax.set_yticks(())
    	ax.set_ylim((0, 1.6))
    	ax.set_xlabel('X')
    	ax.set_ylabel('y')
    	ax.set_xlim(0, 2)
    	fignum += 1

	plt.show()

	"""
	print len(X_train), len(y_train), len(X_test), len(y_test)

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(X_train, y_train)

	# The coefficients
	print('Coefficients: \n', regr.coef_)

	print('Variance score: %.2f' % regr.score(X_test, y_test))

	print "=============================="
	print regr.predict(X_test)
	print y_test
	"""
br = Brew()
docs = br.find_data()
br.copy_data_in_csv(docs)
