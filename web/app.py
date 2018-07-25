from flask import Flask,  Response, request, send_from_directory, jsonify, url_for, session
app = Flask(__name__,template_folder='templates')
app.secret_key = 'xyz'


from flask import render_template
import os

from final import recognize_text_from_image
from classifier import Classifier
# @app.route("/")
# def hello():
#     return "Welcome to Python Flask!"
basedir = os.path.abspath(os.path.dirname(__file__))



 
@app.route("/")
def hello():
    # session['img_classifier'] = Classifier()
    return render_template('index.html')
    # return "Welcome to Python Flask!"

@app.route('/receivedata', methods=['POST'])
def receive_data():
    print(request.form['myData'])
    return "aaa"




@app.route('/uploadajax', methods=['POST'])
def upldfile():
    print("uploadsdadsadas")
    
    if request.method == 'POST':
        print("post")
        files = request.files['file']
        print(files)
        if files:
            # filename = secure_filename(files.filename)
            filename = files.filename
            app.logger.info('FileName: ' + filename)

            updir = os.path.join(basedir, 'upload/')

            final_dir = updir + filename

            fname_after_processing = "static/asd.txt"

            files.save(os.path.join(updir, filename))

            recognize_text_from_image(final_dir, fname_after_processing, Classifier())

            file_size = os.path.getsize(os.path.join(updir, filename))

            # return jsonify({'result_image_location': url_for('txt', filename=fname_after_processing)})
            return jsonify(name=filename, size=file_size, url=  url_for('static', filename= 'asd.txt'))



 
if __name__ == "__main__":
    app.run()




