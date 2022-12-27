from flask import Flask , render_template, request
import final_model

app=Flask(__name__)
final_val=0
final_val1=0
final_val2=0
ans = 0

@app.route("/",methods=["GET","POST"])
def hello():
    global final_val
    global ans
    if request.method=="POST":
        inp=request.form['Yer']
        ans=final_model.memory_prediction(inp)
        # print(ans)
        final_val=ans

    return render_template("index.html", pred_memory = final_val)


if __name__=="__main__":
    app.run(debug=True)