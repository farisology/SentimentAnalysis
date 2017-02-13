#this script will save from stream tweets directly into database
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import sqlite3
import sys

conn = sqlite3.connect('starwars.sqlite')
conn.text_factory = str
cur = conn.cursor()


#consumer key, consumer secret, access token, access secret.
ckey="XXXXXXXXXXXXX"
csecret="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
atoken="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
asecret="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

cur.execute('''DROP TABLE IF EXISTS Tweets ''')
cur.execute('''CREATE TABLE IF NOT EXISTS Tweets
    (id INTEGER PRIMARY KEY, tweet TEXT, username Text, language Text, location Text, twDate Text)''')

class listener(StreamListener):

    def on_data(self, data):

        all_data = json.loads(data)
        tweet = all_data["text"]
        username = all_data["user"]["screen_name"]
        language = all_data["lang"]
        location = all_data["user"]["location"]
        twDate = all_data['created_at']

        cur.execute('INSERT OR IGNORE INTO Tweets (tweet, username, language, location, twDate) VALUES (?, ?, ?, ?, ?)', (tweet, username, language, location, twDate) )

        conn.commit()

        print((username,tweet, language, location, twDate))

        return True

    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["Star Wars", "Rogue One"])
cur.close()
