# bloggie_unwarper_gui.py
# GUI Bloggie (donut) → panorama unwarper with File menu + toolbar buttons.
# Dependencies: pip install opencv-python pillow numpy

import json, math
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

Y_WARP_A, Y_WARP_B, Y_WARP_C = 0.1850, 0.8184, -0.0028
ASPECT = 360.0 / 55.0  # Bloggie default

def build_map(unw_w, unw_h, cx, cy, rmin, rmax, shift=0.0, yA=Y_WARP_A, yB=Y_WARP_B, yC=Y_WARP_C):
    xs = np.arange(unw_w, dtype=np.float32)[None, :]
    ys = np.arange(unw_h, dtype=np.float32)[:, None]
    y = ys / float(unw_h)
    yfrac = yA*(y**2) + yB*y + yC
    radius = (yfrac * (rmax - rmin)) + rmin
    angle  = (0.0 - (xs / float(unw_w)) * (2.0*math.pi)) + shift
    map_x = cx + radius * np.cos(angle)
    map_y = cy + radius * np.sin(angle)
    return map_x.astype(np.float32), map_y.astype(np.float32)

def unwarp(img_bgr, unw_w, cx, cy, rmin, rmax, shift_rad, interp="nearest"):
    unw_h = int(round(unw_w / ASPECT))
    mx, my = build_map(unw_w, unw_h, cx, cy, rmin, rmax, shift_rad)
    mode = cv2.INTER_NEAREST if interp == "nearest" else cv2.INTER_CUBIC
    return cv2.remap(img_bgr, mx, my, interpolation=mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def cv2_to_tk(bgr, fit_w=None, fit_h=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    if fit_w and fit_h:
        pil.thumbnail((fit_w, fit_h), Image.LANCZOS)
    return ImageTk.PhotoImage(pil), pil.size

class BloggieGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bloggie Unwarper (Python GUI)")
        self.geometry("1280x720")
        self.minsize(1000, 620)

        # state
        self.img_path = None
        self.img_bgr  = None
        self.h = self.w = 0

        # params (tk variables)
        self.cx = tk.DoubleVar(value=0.0)
        self.cy = tk.DoubleVar(value=0.0)
        self.rmin = tk.DoubleVar(value=0.0)
        self.rmax = tk.DoubleVar(value=0.0)
        self.pano_w = tk.IntVar(value=1200)
        self.shift_deg = tk.DoubleVar(value=0.0)
        self.interp = tk.StringVar(value="nearest")

        self._pending = None
        self._dragging = False

        self._build_menu()
        self._build_main()
        self.bind("<Key>", self._on_key)

    # ---------- UI ----------
    def _build_menu(self):
        m = tk.Menu(self)
        filem = tk.Menu(m, tearoff=False)
        filem.add_command(label="Open…", command=self.load_image, accelerator="Ctrl+O")
        filem.add_command(label="Save Panorama…", command=self.save_output, accelerator="Ctrl+S")
        filem.add_command(label="Save Params (JSON)…", command=self.save_params, accelerator="Ctrl+P")
        filem.add_separator()
        filem.add_command(label="Exit", command=self.destroy, accelerator="Ctrl+Q")
        m.add_cascade(label="File", menu=filem)
        self.config(menu=m)
        # Shortcuts
        self.bind_all("<Control-o>", lambda e: self.load_image())
        self.bind_all("<Control-s>", lambda e: self.save_output())
        self.bind_all("<Control-p>", lambda e: self.save_params())
        self.bind_all("<Control-q>", lambda e: self.destroy())

    def _build_main(self):
        # canvases
        self.left = tk.Label(self, bg="#222")
        self.right = tk.Label(self, bg="#222")
        self.left.place(x=10, y=10, width=600, height=600)
        self.right.place(x=630, y=10, width=640, height=600)
        for ev, fn in [("<Button-1>", self._start_drag),
                       ("<B1-Motion>", self._drag),
                       ("<ButtonRelease-1>", self._stop_drag)]:
            self.left.bind(ev, fn)

        # bottom controls row (grid instead of hidden-right buttons)
        bar = tk.Frame(self); bar.place(x=10, y=620-5, width=1260, height=90)

        def L(text): return tk.Label(bar, text=text, anchor="w")
        def Spin(var, frm, to, inc, w=8): 
            return tk.Spinbox(bar, from_=frm, to=to, textvariable=var, width=w, increment=inc, command=self.refresh)

        # toolbar buttons (always visible)
        btn_load  = ttk.Button(bar, text="Load image…", command=self.load_image)
        btn_save  = ttk.Button(bar, text="Save panorama…", command=self.save_output)
        btn_param = ttk.Button(bar, text="Save params…", command=self.save_params)

        # layout
        col = 0
        for widget in (btn_load, btn_save, btn_param):
            widget.grid(row=0, column=col, padx=6, pady=3, sticky="w"); col += 1

        # Controls
        widgets = [
            (L("Center X"), Spin(self.cx, 0, 99999, 0.25, 10)),
            (L("Center Y"), Spin(self.cy, 0, 99999, 0.25, 10)),
            (L("Inner R"),  Spin(self.rmin, 0, 99999, 0.5, 8)),
            (L("Outer R"),  Spin(self.rmax, 0, 99999, 0.5, 8)),
            (L("Panorama W"), Spin(self.pano_w, 400, 6000, 20, 8)),
        ]
        for lbl, wid in widgets:
            lbl.grid(row=1, column=col, padx=(14,4), pady=2, sticky="w"); col += 1
            wid.grid(row=1, column=col, padx=(0,8), pady=2, sticky="w"); col += 1

        L("Rotate (°)").grid(row=1, column=col, padx=(10,4), pady=2, sticky="w"); col += 1
        ttk.Scale(bar, from_=0, to=360, variable=self.shift_deg, command=lambda e: self.refresh()
                 ).grid(row=1, column=col, padx=(0,12), pady=2, sticky="we"); col += 1

        L("Interp").grid(row=1, column=col, padx=(6,4), pady=2, sticky="w"); col += 1
        cmb = ttk.Combobox(bar, values=["nearest","cubic"], textvariable=self.interp, width=8, state="readonly")
        cmb.grid(row=1, column=col, padx=(0,8), pady=2, sticky="w")
        cmb.bind("<<ComboboxSelected>>", lambda e: self.refresh())

    # ---------- I/O ----------
    def load_image(self):
        path = filedialog.askopenfilename(
            title="Choose donut image",
            filetypes=[("Images","*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("All","*.*")]
        )
        if not path: return
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", f"Could not open: {path}"); return
        self.img_path = path; self.img_bgr = img; self.h, self.w = img.shape[:2]
        # sensible defaults (Processing sketch)
        self.rmax.set(self.h/2.0 * 0.72)
        self.rmin.set(self.h/2.0 * 0.16)
        self.cx.set(1314.5 if self.w > 1400 else self.w/2.0)
        self.cy.set( 997.25 if self.h > 1000 else self.h/2.0)
        self.pano_w.set(1200); self.shift_deg.set(0.0); self.interp.set("nearest")
        self.refresh(force=True)

    def save_output(self):
        if self.img_bgr is None: return
        pano = self._compute_pano()
        out = filedialog.asksaveasfilename(defaultextension=".jpg",
                    filetypes=[("JPEG","*.jpg"),("PNG","*.png"),("All","*.*")])
        if out:
            cv2.imwrite(out, pano); messagebox.showinfo("Saved", out)

    def save_params(self):
        if self.img_bgr is None: return
        data = dict(
            image_path=self.img_path, w=int(self.w), h=int(self.h),
            cx=float(self.cx.get()), cy=float(self.cy.get()),
            minR=float(self.rmin.get()), maxR=float(self.rmax.get()),
            width=int(self.pano_w.get()), shift_deg=float(self.shift_deg.get()),
            interp=self.interp.get(), yWarp=dict(A=Y_WARP_A,B=Y_WARP_B,C=Y_WARP_C), aspect=ASPECT
        )
        out = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if out: Path(out).write_text(json.dumps(data, indent=2)); messagebox.showinfo("Saved", out)

    # ---------- Interaction ----------
    def _start_drag(self, e): self._dragging = True;  self._update_center_from_click(e)
    def _drag(self, e):       self._update_center_from_click(e) if self._dragging else None
    def _stop_drag(self, e):  self._dragging = False

    def _update_center_from_click(self, e):
        if self.img_bgr is None: return
        disp_w, disp_h = 600, 600
        scale = min(disp_w / self.w, disp_h / self.h)
        img_w, img_h = self.w * scale, self.h * scale
        off_x, off_y = (disp_w - img_w)/2.0, (disp_h - img_h)/2.0
        x = (e.x - off_x) / scale; y = (e.y - off_y) / scale
        x = max(0.0, min(self.w-1, x)); y = max(0.0, min(self.h-1, y))
        self.cx.set(round(x,2)); self.cy.set(round(y,2)); self.refresh()

    def _on_key(self, e):
        if self.img_bgr is None: return
        step = 0.25
        if e.keysym == "Left":  self.cx.set(self.cx.get() - step)
        if e.keysym == "Right": self.cx.set(self.cx.get() + step)
        if e.keysym == "Up":    self.cy.set(self.cy.get() - step)
        if e.keysym == "Down":  self.cy.set(self.cy.get() + step)
        self.refresh()

    # ---------- Render ----------
    def _compute_pano(self):
        return unwarp(self.img_bgr, int(self.pano_w.get()),
                      float(self.cx.get()), float(self.cy.get()),
                      float(self.rmin.get()), float(self.rmax.get()),
                      math.radians(float(self.shift_deg.get())),
                      self.interp.get())

    def _draw_left_overlay(self):
        img = self.img_bgr.copy()
        cx, cy = int(round(self.cx.get())), int(round(self.cy.get()))
        rin, rout = int(round(self.rmin.get())), int(round(self.rmax.get()))
        cv2.circle(img, (cx, cy), 4, (0,0,255), -1)
        cv2.circle(img, (cx, cy), rout, (0,255,0), 2)
        cv2.circle(img, (cx, cy), rin, (255,0,0), 2)
        cv2.line(img, (cx-12, cy), (cx+12, cy), (0,0,255), 1)
        cv2.line(img, (cx, cy-12), (cx, cy+12), (0,0,255), 1)
        return img

    def refresh(self, force=False):
        if not force:
            if self._pending is not None: return
            self._pending = self.after(30, self._refresh_impl)
        else: self._refresh_impl()

    def _refresh_impl(self):
        self._pending = None
        if self.img_bgr is None: return
        left_img = self._draw_left_overlay()
        tk_left, _ = cv2_to_tk(left_img, 600, 600)
        self.left.imgtk = tk_left; self.left.configure(image=tk_left)
        pano = self._compute_pano()
        tk_right, _ = cv2_to_tk(pano, 640, 600)
        self.right.imgtk = tk_right; self.right.configure(image=tk_right)

if __name__ == "__main__":
    BloggieGUI().mainloop()
