
import tkinter as tk 
from tkinter import filedialog
from algo import main
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandas import DataFrame

def UploadAction(event=None):
    filez = filedialog.askopenfilenames(parent=r,title='Choose a file')
    files_data = r.tk.splitlist(filez)
    result = main(files_data)

    import operator
    sorted_d = dict(sorted(result.items(), key=operator.itemgetter(1),reverse=True))

    top_10 = list(sorted_d.keys())[0:10]

    re_insert = dict()
    list_value = list()
    predict_values = list()
        
    for i in top_10:
        list_value.append(i)
        predict_values.append(float(sorted_d[i]))
        re_insert['Algorithms'] = list_value
        re_insert['Accuracy'] = predict_values
    print(re_insert)
    df1 = DataFrame(re_insert,columns=['Algorithms','Accuracy'])

    figure1 = plt.Figure(figsize=(30,15), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.set_title('Best algorithms for your model')
    bar1 = FigureCanvasTkAgg(figure1, r)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    df1 = df1[['Algorithms','Accuracy']].groupby('Algorithms').sum()
    df1.plot(kind='bar', legend=True, ax=ax1)
    

if __name__ == "__main__" : 
    global r
    r = tk.Tk() 
    r.geometry("300x200+10+20")
    r.title('AutoML') 
    r.configure(background = 'light green')
    button = tk.Button(r, text='Open', command=UploadAction)
    button.pack()

    r.mainloop()
