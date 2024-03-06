from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    titles = []
    comments = []

    titles = "Allez vous subir un AVC ?"
    return render_template('index.html', titles = titles, comments = comments) # active_tab='home'

if __name__ == '__main__':
    app.run(debug=True)

# -------------------------------------------------------------------------------------------
