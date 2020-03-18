from flask import Flask,render_template
app=Flask(__name__)

posts = [

        'Hostname of client: ',

        'IP Address: ',

        'CPU usage: ',

        'Memory Usage: ',

        'Disk Space: '
]
f=open("/client1/client.txt",'r')
text=f.read()
t=text.split('newline')
f.close()
f1=open("/client2/client.txt",'r')
text1=f1.read()
t1=text1.split('newline')
f1.close()
@app.route("/")
@app.route("/home")
def home():
	return render_template('home.html', posts=posts , t=t ,t1=t1)



@app.route("/client1")
def client1():
    return render_template('client1.html', t=t)
@app.route("/client2")
def client2():
    return render_template('client2.html', t=t)

if __name__=='__main__':
    app.run("0.0.0.0")
