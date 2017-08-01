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
from sklearn.linear_model import LinearRegression

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
            '2017-07-30', '%Y-%m-%d').date()
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
        print len(docs)
        the_file = open("metric.csv", "w")
        writer = csv.DictWriter(the_file, docs[0].keys())
        writer.writeheader()
        writer.writerows(docs)
        data_df = pd.read_csv("metric.csv")
	data_df['waiting_time'] = data_df.creation_ts - data_df.start_ts
	self.linear_regression(data_df)
	the_file.close()

    def linear_regression(self, data_df):
	lm = LinearRegression()
	new_data_frame = data_df[data_df.columns.difference(["waiting_time", "package_name", "source", "epoch", "version", "owner_name", "nvr", "name", "release", "volume_name", "start_time", "creation_time", "completion_time"])]
	new_data_frame = new_data_frame.fillna(0)
	new_data_frame = new_data_frame.replace("{'image': {'autorebuild': False, 'help':", 1, regex=True)
	for index, row in new_data_frame.iterrows():
	    new_data_frame.loc[index, "completion_ts"] /= (10**9)
	    new_data_frame.loc[index, "start_ts"] /= (10**9)
	    new_data_frame.loc[index, "creation_ts"] /= (10**9)
        print(new_data_frame)
	lm.fit(new_data_frame, data_df.waiting_time)

br = Brew()
docs = br.find_data()
br.copy_data_in_csv(docs)
