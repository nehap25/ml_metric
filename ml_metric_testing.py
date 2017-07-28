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
import matplotlib.pyplot as plt
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
            '2017-07-25', '%Y-%m-%d').date()
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
	# copy builds in csv file
	for build in docs:
	    with open('test.csv', 'a') as csvfile:
        	writer = csv.writer(csvfile, delimiter=',')
		if not build["creation_time"] or not build["start_time"]: 
		    diff = None 
		else: 
		    if build["creation_time"].find(".") != -1:
			created_time = datetime.datetime.strptime(build["creation_time"], '%Y-%m-%d %H:%M:%S.%f')
		    else:
			created_time = datetime.datetime.strptime(build["creation_time"], '%Y-%m-%d %H:%M:%S')
		
		    if build["start_time"].find(".") != -1:
			start_time = datetime.datetime.strptime(build["start_time"], '%Y-%m-%d %H:%M:%S.%f')
	            else:
			start_time = datetime.datetime.strptime(build["start_time"], '%Y-%m-%d %H:%M:%S')
	            diff = str(created_time - start_time)
		
	    	writer.writerow([build["package_name"], build["extra"], build["creation_time"], build["completion_time"], build["package_id"], build["build_id"], build["state"], build["source"], build["epoch"], build["version"], build["completion_ts"], build["owner_id"], build["owner_name"], build["nvr"], build["start_time"], build["creation_event_id"], build["start_ts"], build["volume_id"], build["creation_ts"], build["name"], build["task_id"], build["release"], diff])

	
    def linear_regression(self):
	builds = self.find_data()
	df = pd.DataFrame(builds)
	df.columns = ["package_name", "extra", "creation_time", "completion_time", "package_id", "build_id", "state", "source", "epoch", "version", "completion_ts", "owner_id", "owner_name", "nvr", "start_time", "creation_event_id", "start_ts", "volume_id", "creation_ts", "name", "task_id", "release", "start_creation_diff"]
	lm = LinearRegression()git
	lm.fit(X, builds.start_creation_diff)
	predicted_load = lm.predict(X)
	
	
br = Brew()
docs = br.find_data()
br.copy_data_in_csv(docs)
