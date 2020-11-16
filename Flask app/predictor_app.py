import flask
from flask import redirect, request, url_for, render_template
from predictor_api import make_prediction
from content_api import recommend_by_genre


# Initialize the app

app = flask.Flask(__name__, template_folder='templates')

# An example of routing:
# If they go to the page "/" (this means a GET request
# to the page http://127.0.0.1:5000/), return a simple
# page that says the site is up!

@app.route("/", methods=["GET", "POST"])
def hello():
    return render_template('index.html')


@app.route("/collaborative", methods=["GET", "POST"])
def collaborative():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    # if request.method == "POST":

    prediction = None
    if request.method == 'POST':
        favorite_movie = request.form['favorite_movie']
        prediction = make_prediction(favorite_movie)

    # show the form, it wasn't submitted
    return render_template('collaborative.html', prediction=prediction)


@app.route("/content", methods=["GET", "POST"])
def content():
    # request.args contains all the arguments passed by our form
    # comes built in with flask. It is a dictionary of the form
    # "form name (as set in template)" (key): "string in the textbox" (value)
    # if request.method == "POST":

    prediction = None
    if request.method == 'POST':
        favorite_movie = request.form['favorite_movie']
        prediction = recommend_by_genre(favorite_movie)

    # show the form, it wasn't submitted
    return render_template('content.html', prediction=prediction)

# Start the server, continuously listen to requests.
# We'll have a running web app!

if __name__=="__main__":
    # For local development:
    # app.run(debug=True)
    app.run(debug=True)

    # For public web serving:
    # app.run(host='0.0.0.900')
