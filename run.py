from flask import Flask,render_template
from mainprogramfor5_12 import sepal_len,sepal_wid,petal_len,petal_wid
app=Flask("__name__")

sepal_len1=sepal_len
sepal_wid1=sepal_wid
petal_len1=petal_len
petal_wid1=petal_wid

@app.route('/')
def main():

	return render_template('home.html',sepal_len1,sepal_wid1,petal_len1,petal_wid1)

if __name__=='__main__':
	app.run(debug=True)
