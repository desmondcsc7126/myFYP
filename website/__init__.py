from flask import Flask

def createApp():
    
    app = Flask(__name__, template_folder ='template')
    app.config['SECRET_KEY'] = "desmondcsc1234"

    from .page import page
    from .auth import auth

    app.register_blueprint(page,url_prefix = '/')
    app.register_blueprint(auth,url_prefix = '/')

    return app