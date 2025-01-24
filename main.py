from tkinter import filedialog
from tkinter.simpledialog import askstring
import tkinter as tk
from PIL import Image

root = tk.Tk()
root.title("PNG to ICO Converter Second Edition")
root.geometry("400x160")

def png_select():
    global file_path
    file_path = filedialog.askopenfilename(title="Select a PNG File", filetypes=[("PNG Dosyaları", "*.png"), ("Tüm Dosyalar", "*.*")])

def convert():
    global img
    img = Image.open(file_path)

def ico_download():
    if img:
        save_path = filedialog.asksaveasfilename(defaultextension=".ico", filetypes=[("ICO Files", "*.ico")])
        if save_path:
            img.save(save_path, format="ICO")

select_png_button = tk.Button(root, text="Select a PNG file", command=png_select)
select_png_button.pack(pady=10)

convert_file_button = tk.Button(root, text="Convert the PNG file", command=convert)
convert_file_button.pack(pady=10)

download_ico_button = tk.Button(root, text="Download the ico file", command=ico_download)
download_ico_button.pack(pady=10)

Creator = tk.Label(root, text="Created By Kuzey Kaan Özyiğit")
Creator.pack()

root.mainloop()