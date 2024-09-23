import gradio as gr
import sys

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def isatty(self):
        return False    

sys.stdout = Logger("output.log")

def test(x):
    print("This is a test")
    print(f"Your function is running with input {x}...")
    return x

def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()

with gr.Blocks() as demo:
    with gr.Row():
        input = gr.Textbox()
        output = gr.Textbox()
    btn = gr.Button("Run")
    btn.click(test, input, output)
    
    logs = gr.Textbox()
    demo.load(read_logs, None, logs, every=1)
    
demo.queue().launch()