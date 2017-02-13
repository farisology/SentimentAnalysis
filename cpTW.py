import sys
import sqlite3
import time
import ssl
import urllib
from urlparse import urljoin
from urlparse import urlparse
import re
from datetime import datetime, timedelta
import string
import zlib

# Open the raw data retrieved from the network
conn = sqlite3.connect('starwarsL4.sqlite')
conn.text_factory = str
cur = conn.cursor()

# read only the english tweets and split words to count frequency
lan = ('en',)
counts = dict()
for row in cur.execute('SELECT * FROM Tweets WHERE language =?', lan):
    text = row[1]
    text = text.translate(None, string.punctuation)
    text = text.translate(None, '1234567890')
    text = text.strip()
    text = text.lower()
    words = text.split()
    #print text
    for word in words:
        if len(word) < 4 : continue
        counts[word] = counts.get(word,0) + 1
        
# Find the top 100 words
words = sorted(counts, key=counts.get, reverse=True)
highest = None
lowest = None
for w in words[:100]:
    if highest is None or highest < counts[w] :
        highest = counts[w]
    if lowest is None or lowest > counts[w] :
        lowest = counts[w]
print 'Range of counts:',highest,lowest

# Spread the font sizes across 20-100 based on the count
bigsize = 80
smallsize = 20

fhand = open('gword.js','w')
fhand.write("gword = [")
first = True
for k in words[:100]:
    if not first : fhand.write( ",\n")
    first = False
    size = counts[k]
    size = (size - lowest) / float(highest - lowest)
    size = int((size * bigsize) + smallsize)
    fhand.write("{text: '"+k+"', size: "+str(size)+"}")
fhand.write( "\n];\n")

print "Output written to gword.js"
print "Open gword.htm in a browser to view"