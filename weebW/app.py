from flask import Flask, request, render_template

app = Flask(__name__)


@app.route('/', methods = ['POST', 'GET'])
def index():
    if request.method == "POST":
        return render_template('answer.html')

    else:
        return render_template("index.html")


@app.route('/answer/<characterName>')


if __name__ == '__main__':
    app.run()
