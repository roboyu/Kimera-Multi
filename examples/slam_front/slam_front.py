import tkinter as tk
from PIL import Image, ImageTk
import xml.etree.ElementTree as ET
import os

background_photo = '/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/examples/slam_front/background.jpeg'
launch_file = '/media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/launch/kimera_vio_jackal.launch'
run_commands = {"default": "cd /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/examples;bash run.sh",
                "run single": "cd /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/examples;bash run.sh",
                "run multi": "cd /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/examples;bash run_multi_times.sh",
                "run real": "cd /media/sysu/new_volume1/80G/sysu/herh/kimera_multi_ws/src/kimera_multi/examples;bash run_real.sh"}


class MyFrame():
    def __init__(self, image_path, run_commands):
        self.root = tk.Tk()
        self.root.title("SLAM")
        self.root.geometry("600x400+0+0")
        self.root.resizable(False, False)
        self.root.config(bg="lightgreen")
        self.image_path = image_path
        # img = Image.open(image_path)
        # img = img.resize((600, 400), Image.Resampling.LANCZOS)
        # self.background = ImageTk.PhotoImage(img)
        # tk.Label(self.root, image=self.background).pack()

        self.runCommands = run_commands
        self.runCommand = run_commands["default"]

    def start(self):
        self.add_compoent()
        self.root.mainloop()

    def change_parameters(self):
        tree = ET.parse(launch_file)
        r = tree.getroot()
        Labels_text = []
        Entry_content = []
        keys = []
        for param in r.findall('arg'):
            if param.get('value') != None:
                Labels_text.append(param.get('name'))
                Entry_content.append(param.get('value'))
                keys.append("value")
            elif param.get('default') != None:
                Labels_text.append(param.get('name'))
                Entry_content.append(param.get('default'))
                keys.append("default")
        # for i in range(len(Labels_text)):
        #     print("{} : {}".format(Labels_text[i],Entry_content[i]))
        n = len(Labels_text)

        self.button_disabled()

        def close():
            for i in range(n):
                Entry_content[i] = Entrys[i].get()
                for arg in r.findall('arg'):
                    if arg.get('name') == Labels_text[i]:
                        arg.set(keys[i], Entry_content[i])
                        break
            tree.write(launch_file, encoding='utf-8', xml_declaration=True)
            self.button_normal()
            root.destroy()

        # n=30#the number of parameter
        max_row = 18  # the number of max row
        root = tk.Tk()
        root.protocol("WM_DELETE_WINDOW", close)
        root.title("调整参数")
        root.config(bg="lightgreen")
        Labels = []
        Entrys = []
        row = max_row
        if n < max_row:
            row = n
        col = (n+max_row-1)//max_row  # the number of colume
        root.geometry(str(250*col-15)+"x"+str(30*row+50)+"+100+100")
        root.resizable(False, False)
        # Labels_text=["helo"]*n
        # Entry_content=["12345"]*n
        for i in range(n):
            k = i//row
            j = i % row
            if len(Labels_text[i]) > 8:
                con = Labels_text[i][:8]
            else:
                con = Labels_text[i]
            label = tk.Label(root, text=con, width=8, bg="lightblue")
            label.place(x=250*k+10, y=j*30+10)
            entry = tk.Entry(root, width=15, bg="lightblue")
            if Entry_content[i] != None:
                con = Entry_content[i]
            else:
                con = "None"
            entry.insert(0, con)
            entry.place(x=250*k+100, y=j*30+10)
            Labels.append(label)
            Entrys.append(entry)
        tk.Button(root, text="确定", bg="lightblue", command=close).place(
            x=250*col-77, y=(row*30+10))
        root.mainloop()

    def select_mode(self):
        root1 = tk.Tk()
        root1.title("选择模式")
        root1.geometry("600x"+str(40*len(self.runCommands)+200)+"+100+100")
        root1.resizable(False, False)
        root1.config(bg="lightgreen")
        tk.Label(root1, text="可选模式：", font=("Arial", 15, "bold"),
                 bg="lightblue").place(x=100, y=80)
        i = 0
        for k in self.runCommands.keys():
            tk.Label(root1, text=k, font=("Arial", 15, "bold"),
                     bg="lightblue").place(x=100, y=120+40*i)
            i += 1
        tk.Label(root1, text="你的模式: ", font=("Arial", 15, "bold"),
                 bg="lightblue").place(x=300, y=40+20*len(self.runCommands))
        entry = tk.Entry(root1, font=("Arial", 15, "bold"), bg="lightblue")
        entry.place(x=300, y=80+20*len(self.runCommands))

        def close():
            content = entry.get()
            if content in self.runCommands.keys():
                self.runCommand = self.runCommands[content]
            print(self.runCommand)
            self.button_normal()
            root1.destroy()
        tk.Button(root1, text="确定", bg="lightblue",
                  command=close).place(x=550, y=170+i*40)
        root1.mainloop()

    def run(self):
        self.button_disabled()
        os.system(self.runCommand)
        self.button_normal()

    def add_compoent(self):
        custom_font = ("Arial", 13, "bold")
        self.button_parameters = tk.Button(
            self.root, text="调整参数", font=custom_font, bg="lightblue", command=self.change_parameters)
        self.button_parameters.place(x=200, y=200)
        self.button_select = tk.Button(
            self.root, text="选择模式", font=custom_font, bg="lightblue", command=self.select_mode)
        self.button_select.place(x=310, y=200)
        self.button_run = tk.Button(self.root, text="运行", font=(
            "Arial", 15, "bold"), bg="lightblue", command=self.run)
        self.button_run.place(x=270, y=140)

    def button_disabled(self):
        self.button_parameters.config(state='disabled')
        self.button_run.config(state='disabled')
        self.button_select.config(state="disabled")

    def button_normal(self):
        self.button_parameters.config(state='normal')
        self.button_run.config(state='normal')
        self.button_select.config(state="normal")


my_frame = MyFrame(background_photo, run_commands)
my_frame.start()
