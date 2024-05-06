import speech_recognition as sr
import pyttsx3
from llm_util import LLM
from flask import Flask, render_template, request

app = Flask(__name__)
llm_obj = LLM()

class SpeechToText:
    def __init__(self, lang='en'):
        self.r = sr.Recognizer()
        self.language = lang
        
    def extract_text_from_speech(self):
        with sr.Microphone() as source2:
            print("Listening....")
            self.r.adjust_for_ambient_noise(source2, duration=0.3)
            audio2 = self.r.listen(source2)
            try:
                MyText = self.r.recognize_google(audio2)
                MyText = MyText.lower()
                return MyText
            except sr.UnknownValueError:
                return "Sorry, I couldn't understand what you said."
            except sr.RequestError as e:
                return f"Sorry, there was an error with the speech recognition service: {e}"

class TextToSpeech:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.is_engine_running = False
        
    def text_to_speech(self, command):
        if not self.is_engine_running:
            self.engine.startLoop(False)
            self.is_engine_running = True
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)
        self.engine.say(command)
        self.engine.iterate()  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_query', methods=['POST'])
def process_query():
    if request.method == 'POST':
        query = request.form['query']
        response = llm_obj.answer_to_the_question(query)
        tts = TextToSpeech()
        tts.text_to_speech(response)
        return render_template('index.html', response=response)

@app.route('/speech_to_text', methods=['GET'])
def speech_to_text():
    stt = SpeechToText()
    query = stt.extract_text_from_speech()
    response = llm_obj.answer_to_the_question(query)
    tts = TextToSpeech()
    tts.text_to_speech(response)
    return render_template('index.html', query=query, response=response)

if __name__ == '__main__':
    app.run(debug=True)
