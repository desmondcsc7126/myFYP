from website import createApp

application = createApp()

if __name__ == '__main__':
    application.run(debug=True)
    # application.run(host='0.0.0.0', port=80, debug=True)