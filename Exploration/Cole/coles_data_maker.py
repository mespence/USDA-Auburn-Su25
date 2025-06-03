# the actual data maker is complicated, so this thing is going to add values to a csv so we can test whether the live reader works
# also this gives me the ability to get back in the swing of things
import csv
import random
from time import sleep
import os

outfilename = "testdata1.csv"
counter = 0

with open(outfilename, "a", newline="", buffering=1) as csvfile:
    writer = csv.writer(csvfile)
    while True:
        writer.writerow([counter, random.randint(0, 100)])  # writes a random integer between 0 and 100
        csvfile.flush()  # ensure data is written immediately
        os.fsync(csvfile.fileno())  # force write to disk
        counter += 1
        sleep(0.1)
