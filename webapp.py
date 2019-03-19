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

from joblib import load
import constant

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


def messageReceived(methods=['GET', 'POST']):
    print('message was received!!!')


@socketio.on('my event')
def handle_my_custom_event(msg, methods=['GET', 'POST']):

    toxicityClassifier = ToxicityClassifier()

    print('received my event: username:' + msg['username'] + ', message:' + msg['message'] )
    print('chat toxicity analyzed...' + str(toxicityClassifier.isToxic( msg['message'] )))
    if toxicityClassifier.isToxic(msg['message']) == True:
        socketio.emit('my response', { 'username': msg['username'], 'message': msg['message'], 'is_toxic': '1'}, callback=messageReceived)
    else:
        socketio.emit('my response', { 'username': msg['username'], 'message': msg['message'], 'is_toxic': '0'}, callback=messageReceived)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=True, host='0.0.0.0', port=port)

    print("application has started.")
