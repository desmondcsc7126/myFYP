from flask import Blueprint, render_template

page = Blueprint('page',__name__)

@page.route('/')
def homepage():
    return render_template('base.html')