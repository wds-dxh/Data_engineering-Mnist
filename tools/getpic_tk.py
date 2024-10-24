'''
Author: wds-win11 wdsnpshy@163.com
Date: 2024-10-23 12:42:28
LastEditors: wds-wsl_ubuntu22 wdsnpshy@163.com
LastEditTime: 2024-10-24 01:36:56
FilePath: /Data_engineering/tools/getpic_tk.py
Description: 分割脚本
微信: 15310638214 
邮箱：wdsnpshy@163.com 
Copyright (c) 2024 by ${wds-win11}, All Rights Reserved. 
'''
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ImageSplitter:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        self.canvas = tk.Canvas(window, width=800, height=600)
        self.canvas.pack()

        self.btn_load = tk.Button(window, text="加载图片", command=self.load_image)
        self.btn_load.pack(side="left")

        self.btn_split = tk.Button(window, text="分割图片", command=self.split_image)
        self.btn_split.pack(side="left")
        self.btn_split.config(state="disabled")

        self.image = None
        self.tk_image = None
        self.points = []
        self.dragging = False
        self.drag_start = None
        self.grid_lines = []

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image()
            self.btn_split.config(state="normal")
            self.points = []
            self.clear_grid()

    def display_image(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        self.ratio = min(800/width, 600/height)
        new_size = (int(width * self.ratio), int(height * self.ratio))
        image = cv2.resize(image, new_size)
        self.tk_image = ImageTk.PhotoImage(Image.fromarray(image))
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

    def on_click(self, event):
        x, y = event.x, event.y
        if len(self.points) < 2:
            self.points.append((x, y))
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", tags="point")
            if len(self.points) == 2:
                self.draw_rectangle()
                self.draw_grid()
        else:
            # Check if click is near any point
            for i, (px, py) in enumerate(self.points):
                if abs(x - px) < 5 and abs(y - py) < 5:
                    self.dragging = True
                    self.drag_start = (x, y)
                    self.drag_point = i
                    break

    def on_drag(self, event):       # 拖动
        if self.dragging:
            x, y = event.x, event.y
            dx = x - self.drag_start[0]
            dy = y - self.drag_start[1]
            self.points[self.drag_point] = (x, y)
            self.canvas.move(f"point{self.drag_point}", dx, dy)
            self.drag_start = (x, y)
            self.draw_rectangle()
            self.draw_grid()

    def on_release(self, event):
        self.dragging = False

    def draw_rectangle(self):   # 画矩形
        self.canvas.delete("rectangle")
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", tags="rectangle")

    def draw_grid(self):    # 画网格
        self.clear_grid()
        if len(self.points) == 2:
            x1, y1 = min(self.points[0][0], self.points[1][0]), min(self.points[0][1], self.points[1][1])
            x2, y2 = max(self.points[0][0], self.points[1][0]), max(self.points[0][1], self.points[1][1])
            width = x2 - x1
            height = y2 - y1
            cell_width = width / 27  # 修改为27
            cell_height = height / 38  # 修改为38

            for i in range(1, 27):  # 修改为27
                x = x1 + i * cell_width
                line = self.canvas.create_line(x, y1, x, y2, fill="blue", tags="grid")
                self.grid_lines.append(line)

            for i in range(1, 38):  # 修改为38
                y = y1 + i * cell_height
                line = self.canvas.create_line(x1, y, x2, y, fill="blue", tags="grid")
                self.grid_lines.append(line)

    def clear_grid(self):   # 清除网格
        for line in self.grid_lines:
            self.canvas.delete(line)
        self.grid_lines = []

    def split_image(self):          #     分割图片
        if len(self.points) != 2:
            messagebox.showerror("错误", "请选择两个对角点")
            return

        output_folder = "split_images"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        x1, y1 = min(self.points[0][0], self.points[1][0]), min(self.points[0][1], self.points[1][1])
        x2, y2 = max(self.points[0][0], self.points[1][0]), max(self.points[0][1], self.points[1][1])

        x1, y1 = int(x1 / self.ratio), int(y1 / self.ratio)
        x2, y2 = int(x2 / self.ratio), int(y2 / self.ratio)

        roi = self.image[y1:y2, x1:x2]

        cell_width = (x2 - x1) / 27  # 修改为27
        cell_height = (y2 - y1) / 38  # 修改为38

        for i in range(38):  # 修改为38
            for j in range(27):  # 修改为27
                start_y = int(i * cell_height)
                start_x = int(j * cell_width)
                end_y = int((i + 1) * cell_height)
                end_x = int((j + 1) * cell_width)
                
                cell = roi[start_y:end_y, start_x:end_x]
                # 处理一下，把块的边框去掉，向内缩小10个像素，避免边框的干扰,然后用白色填充
                cell = cell[2:-2, 2:-2]
                cell = cv2.copyMakeBorder(cell, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                output_path = os.path.join(output_folder, f"cell_{i}_{j}.png")
                cv2.imwrite(output_path, cell)

        messagebox.showinfo("完成", f"图片已分割并保存到 {output_folder} 文件夹")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSplitter(root, "图片分割器")
    root.mainloop()
