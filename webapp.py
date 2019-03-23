# following three lines should have come first than any other imports.
import gevent.monkey
gevent.monkey.patch_all()

import sys
import json
import os
import gevent


from flask import Flask, render_template
from flask_socketio import SocketIO

from detox_engine import ToxicityClassifier

import irc.bot
import requests
import threading

toxicityClassifier = ToxicityClassifier()

class TwitchBot(irc.bot.SingleServerIRCBot):
    def __init__(self, username, client_id, token, channel):
        self.client_id = client_id
        self.token = token
        self.channel = '#' + channel

        # create a text file using channel name that bot is running on
        self.log = open(channel + ".txt", "a")

        # Get the channel id, we will need this for v5 API calls
        url = 'https://api.twitch.tv/kraken/users?login=' + channel
        headers = {'Client-ID': client_id, 'Accept': 'application/vnd.twitchtv.v5+json'}
        r = requests.get(url, headers=headers).json()
        self.channel_id = r['users'][0]['_id']

        # Create IRC bot connection
        server = 'irc.chat.twitch.tv'
        port = 6667
        print('Connecting to ' + server + ' on port ' + str(port) + '...')
        irc.bot.SingleServerIRCBot.__init__(self, [(server, port, 'oauth:' + token)], username, username)

        # load toxic engine
        self.toxicityClassifier = ToxicityClassifier()

    def on_welcome(self, c, e):
        print('Joining ' + self.channel)

        # You must request specific capabilities before you can use them
        c.cap('REQ', ':twitch.tv/membership')
        c.cap('REQ', ':twitch.tv/tags')
        c.cap('REQ', ':twitch.tv/commands')
        c.join(self.channel)

    def on_pubmsg(self, c, e):

        # If a chat message starts with an exclamation point, try to run it as a command
        if e.arguments[0][:1] == '!':
            # following 3 lines are disabled
            cmd = e.arguments[0].split(' ')[0][1:]
            # print('Received command: ' + cmd)
            # self.do_command(e, cmd)
        else:
            self.log.write(e.arguments[0] + "\n")
            log_to_write = ""

            if self.toxicityClassifier.isToxic(e.arguments[0]) == True:
                log_to_write = "***TOXIC CHAT FOUND ***" + e.arguments[0] + "\n"
                print(">[toxic] msg:" + e.arguments[0])
                socketio.emit('my response',
                              {'username': 'Twitchbot', 'message': e.arguments[0], 'is_toxic': '1'},
                              callback=messageReceived)
                # Enable below line if you want to send private warning message to toxic chatter
                # c.privmsg(self.channel, "Stop using toxic chat!!!")
            else:
                log_to_write = e.arguments[0] + "\n"
                print(">msg:" + e.arguments[0])
                socketio.emit('my response',
                              {'username': 'Twitchbot', 'message': e.arguments[0], 'is_toxic': '0'},
                              callback=messageReceived)

            self.log.write(log_to_write)
            self.log.flush()
        return

    # these are for genenral purpose, you won't need it for chat toxicity check but leave it as it is just in case
    def do_command(self, e, cmd):
        c = self.connection

        # Poll the API to get current game.
        if cmd == "game":
            url = 'https://api.twitch.tv/kraken/channels/' + self.channel_id
            headers = {'Client-ID': self.client_id, 'Accept': 'application/vnd.twitchtv.v5+json'}
            r = requests.get(url, headers=headers).json()
            # c.privmsg(self.channel, r['display_name'] + ' is currently playing ' + r['game'])

        # Poll the API the get the current status of the stream
        elif cmd == "title":
            url = 'https://api.twitch.tv/kraken/channels/' + self.channel_id
            headers = {'Client-ID': self.client_id, 'Accept': 'application/vnd.twitchtv.v5+json'}
            r = requests.get(url, headers=headers).json()
            # c.privmsg(self.channel, r['display_name'] + ' channel title is currently ' + r['status'])

        # Provide basic information to viewers for specific commands
        elif cmd == "raffle":
            message = "This is an example bot, replace this text with your raffle text."
            # c.privmsg(self.channel, message)
        elif cmd == "schedule":
            message = "This is an example bot, replace this text with your schedule text."
            # c.privmsg(self.channel, message)

        # The command was not recognized
        else:
            # c.privmsg(self.channel, "Did not understand command: " + cmd)
            print("Did not understand command:" + cmd)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asdf1234'
socketio = SocketIO(app)

@app.before_first_request
def load_module():
    print("initiating before first request...")
    # do nothing for now

@app.route('/', methods=['GET', 'POST'] )
def sessions():
    return render_template('session.html')

@app.route('/chat', methods=['GET', 'POST'] )
def chats():
    return render_template('chat.html')

@app.route('/twitch', methods=['GET', 'POST'] )
def twitchsessions():
    return render_template('chat.html')

def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')


@socketio.on('my event')
def handle_my_custom_event(msg, methods=['GET', 'POST']):

    print('received my event: username:' + msg['username'] + ', message:' + msg['message'] )
    print('chat toxicity analyzed...' + str(toxicityClassifier.isToxic( msg['message'] )))
    if toxicityClassifier.isToxic(msg['message']) == True:
        socketio.emit('my response', { 'username': msg['username'], 'message': msg['message'], 'is_toxic': '1'}, callback=messageReceived)
    else:
        socketio.emit('my response', { 'username': msg['username'], 'message': msg['message'], 'is_toxic': '0'}, callback=messageReceived)

def main():

    if not sys.version_info[:1] == (3,):
        print(sys.version_info[:1] )
        sys.stderr.write("Python version 3 is required.\n")
        exit(1)

    if len(sys.argv) != 5:
        print("Usage: webapp <username> <client id> <token> <channel>")
        sys.exit(1)

    username  = sys.argv[1]
    client_id = sys.argv[2]
    token     = sys.argv[3]
    channel   = sys.argv[4]

    bot = TwitchBot(username, client_id, token, channel)
    t = threading.Thread(target=bot.start)
    t.start()

    print("Twitchbot has started.")

    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=True, host='0.0.0.0', port=port)

    print("Web application has started.")

if __name__ == "__main__":
    main()
