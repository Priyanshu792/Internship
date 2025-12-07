import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import ttkbootstrap as tb
from tkinter.scrolledtext import ScrolledText
import numpy as np
from typing import Optional, Tuple


def parse_matrix(text: str) -> np.ndarray:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Input is empty")
    rows = []
    for ln in lines:
        parts = [p for p in ln.replace(',', ' ').split() if p]
        try:
            row = [float(p) for p in parts]
        except ValueError:
            raise ValueError(f"Non-numeric value found in line: '{ln}'")
        rows.append(row)
    lengths = {len(r) for r in rows}
    if len(lengths) != 1:
        raise ValueError("Rows have inconsistent number of columns")
    return np.array(rows)


def matrix_to_text(mat: np.ndarray) -> str:
    lines = []
    for r in mat:
        lines.append(' '.join(str(x) for x in r))
    return '\n'.join(lines)


class MatrixToolApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Matrix Operations Tool")

        self._create_menu()

        # Main layout: inputs on left, results on right
        main = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=420)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=1)

        # Inputs
        inp_label = ttk.Label(left, text="Matrices (paste rows; values sep by space/comma)", font=(None, 10, 'bold'))
        inp_label.pack(anchor=tk.W, padx=6, pady=(6, 0))

        self.a_frame = ttk.LabelFrame(left, text='Matrix A')
        self.a_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.a_txt = ScrolledText(self.a_frame, height=10)
        self.a_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.b_frame = ttk.LabelFrame(left, text='Matrix B')
        self.b_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0,6))
        self.b_txt = ScrolledText(self.b_frame, height=10)
        self.b_txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Toolbar below inputs
        toolbar = ttk.Frame(left)
        toolbar.pack(fill=tk.X, padx=6, pady=(0,6))

        tb.Button(toolbar, text="Add A + B", bootstyle='success', command=self.add).pack(side=tk.LEFT, padx=4)
        tb.Button(toolbar, text="A - B", bootstyle='info', command=self.sub).pack(side=tk.LEFT, padx=4)
        tb.Button(toolbar, text="A × B", bootstyle='primary', command=self.mul).pack(side=tk.LEFT, padx=4)

        self.single_choice = tk.StringVar(value='A')
        ttk.Radiobutton(toolbar, text='Use A', variable=self.single_choice, value='A').pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(toolbar, text='Use B', variable=self.single_choice, value='B').pack(side=tk.LEFT)

        tb.Button(toolbar, text='Transpose', bootstyle='secondary', command=self.transpose).pack(side=tk.LEFT, padx=6)
        tb.Button(toolbar, text='Determinant', bootstyle='warning', command=self.determinant).pack(side=tk.LEFT, padx=4)

        tb.Button(toolbar, text='Example', bootstyle='light', command=self.fill_example).pack(side=tk.RIGHT, padx=4)
        tb.Button(toolbar, text='Clear', bootstyle='light', command=self.clear_all).pack(side=tk.RIGHT)

        # Right: result preview
        res_label = ttk.Label(right, text="Result", font=(None, 10, 'bold'))
        res_label.pack(anchor=tk.W, padx=6, pady=(6, 0))

        self.result_frame = ttk.Frame(right)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Treeview for matrix-like results
        self.tree = None
        self._create_result_area()

        # bottom actions
        bottom = ttk.Frame(right)
        bottom.pack(fill=tk.X, padx=6, pady=(0,6))
        tb.Button(bottom, text='Copy Result', bootstyle='secondary', command=self.copy_result).pack(side=tk.LEFT)
        tb.Button(bottom, text='Save Result as CSV', bootstyle='secondary', command=self.save_result_csv).pack(side=tk.LEFT, padx=6)

        # status bar
        self.status = tk.StringVar(value='Ready')
        status_bar = ttk.Label(root, textvariable=self.status, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ---------- UI helpers ----------
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        filem = tk.Menu(menubar, tearoff=0)
        filem.add_command(label='Load Matrix A...', command=lambda: self.load_csv_into('A'))
        filem.add_command(label='Load Matrix B...', command=lambda: self.load_csv_into('B'))
        filem.add_separator()
        filem.add_command(label='Exit', command=self.root.quit)
        menubar.add_cascade(label='File', menu=filem)

        helpm = tk.Menu(menubar, tearoff=0)
        helpm.add_command(label='About', command=self._about)
        menubar.add_cascade(label='Help', menu=helpm)

        self.root.config(menu=menubar)

    def _create_result_area(self):
        # clear frame
        for w in self.result_frame.winfo_children():
            w.destroy()

        self.result_message = ttk.Label(self.result_frame, text='No result yet', anchor=tk.CENTER)
        self.result_message.pack(fill=tk.BOTH, expand=True)
        self.current_result = None

    def _show_matrix_result(self, mat: np.ndarray):
        # display matrix in treeview
        for w in self.result_frame.winfo_children():
            w.destroy()

        rows, cols = mat.shape
        tree = ttk.Treeview(self.result_frame, columns=[f'c{i}' for i in range(cols)], show='headings')
        for i in range(cols):
            tree.heading(f'c{i}', text=f'Col {i}')
            tree.column(f'c{i}', width=80, anchor='center')

        for r in mat:
            tree.insert('', tk.END, values=[self._fmt(x) for x in r])

        tree.pack(fill=tk.BOTH, expand=True)
        self.tree = tree
        self.current_result = mat
        self.status.set(f'Result: {rows}×{cols} matrix')

    def _show_scalar_result(self, value: float, label: str = 'Result'):
        for w in self.result_frame.winfo_children():
            w.destroy()
        lbl = ttk.Label(self.result_frame, text=f"{label}: {self._fmt(value)}", anchor=tk.CENTER, font=(None, 12))
        lbl.pack(fill=tk.BOTH, expand=True)
        self.current_result = value
        self.status.set(f'{label} shown')

    def _fmt(self, x):
        # pretty format for numbers
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.6g}"

    # ---------- operations ----------
    def _get_matrices(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        a_str = self.a_txt.get('1.0', tk.END)
        b_str = self.b_txt.get('1.0', tk.END)
        a = b = None
        if a_str.strip():
            a = parse_matrix(a_str)
        if b_str.strip():
            b = parse_matrix(b_str)
        return a, b

    def _handle_error(self, err: Exception):
        messagebox.showerror('Error', str(err))
        self.status.set(f'Error: {err}')

    def add(self):
        try:
            a, b = self._get_matrices()
            if a is None or b is None:
                raise ValueError('Both matrices A and B are required for addition')
            if a.shape != b.shape:
                raise ValueError('Matrices must have the same shape for addition')
            res = a + b
            self._show_matrix_result(res)
        except Exception as e:
            self._handle_error(e)

    def sub(self):
        try:
            a, b = self._get_matrices()
            if a is None or b is None:
                raise ValueError('Both matrices A and B are required for subtraction')
            if a.shape != b.shape:
                raise ValueError('Matrices must have the same shape for subtraction')
            res = a - b
            self._show_matrix_result(res)
        except Exception as e:
            self._handle_error(e)

    def mul(self):
        try:
            a, b = self._get_matrices()
            if a is None or b is None:
                raise ValueError('Both matrices A and B are required for multiplication')
            if a.shape[1] != b.shape[0]:
                raise ValueError('Inner dimensions must match for multiplication')
            res = a.dot(b)
            self._show_matrix_result(res)
        except Exception as e:
            self._handle_error(e)

    def transpose(self):
        try:
            choice = self.single_choice.get()
            a, b = self._get_matrices()
            mat = a if choice == 'A' else b
            if mat is None:
                raise ValueError(f'Matrix {choice} is empty')
            res = mat.T
            self._show_matrix_result(res)
        except Exception as e:
            self._handle_error(e)

    def determinant(self):
        try:
            choice = self.single_choice.get()
            a, b = self._get_matrices()
            mat = a if choice == 'A' else b
            if mat is None:
                raise ValueError(f'Matrix {choice} is empty')
            if mat.shape[0] != mat.shape[1]:
                raise ValueError('Determinant requires a square matrix')
            det = float(np.linalg.det(mat))
            self._show_scalar_result(det, label='Determinant')
        except Exception as e:
            self._handle_error(e)

    # ---------- utilities ----------
    def fill_example(self):
        ex_a = "1 2 3\n4 5 6\n7 8 9"
        ex_b = "9 8 7\n6 5 4\n3 2 1"
        self.a_txt.delete('1.0', tk.END)
        self.b_txt.delete('1.0', tk.END)
        self.a_txt.insert(tk.END, ex_a)
        self.b_txt.insert(tk.END, ex_b)
        self.status.set('Example matrices filled')

    def clear_all(self):
        self.a_txt.delete('1.0', tk.END)
        self.b_txt.delete('1.0', tk.END)
        self._create_result_area()
        self.status.set('Cleared')

    def copy_result(self):
        if self.current_result is None:
            messagebox.showinfo('Info', 'No result to copy')
            return
        if isinstance(self.current_result, np.ndarray):
            txt = matrix_to_text(self.current_result)
        else:
            txt = str(self.current_result)
        self.root.clipboard_clear()
        self.root.clipboard_append(txt)
        self.status.set('Result copied to clipboard')

    def save_result_csv(self):
        if self.current_result is None:
            messagebox.showinfo('Info', 'No result to save')
            return
        filename = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')])
        if not filename:
            return
        try:
            if isinstance(self.current_result, np.ndarray):
                np.savetxt(filename, self.current_result, delimiter=',', fmt='%g')
            else:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(str(self.current_result))
            self.status.set(f'Saved result to {filename}')
        except Exception as e:
            self._handle_error(e)

    def load_csv_into(self, which: str):
        filename = filedialog.askopenfilename(filetypes=[('CSV', '*.csv'), ('All files', '*.*')])
        if not filename:
            return
        try:
            mat = np.loadtxt(filename, delimiter=',')
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            txt = matrix_to_text(mat)
            if which == 'A':
                self.a_txt.delete('1.0', tk.END)
                self.a_txt.insert(tk.END, txt)
            else:
                self.b_txt.delete('1.0', tk.END)
                self.b_txt.insert(tk.END, txt)
            self.status.set(f'Loaded {which} from {filename}')
        except Exception as e:
            self._handle_error(e)

    def _about(self):
        messagebox.showinfo('About', 'Matrix Operations Tool\nImproved UI using ttk\nSupports CSV load/save')


def main():
    root = tb.Window(themename='darkly')
    app = MatrixToolApp(root)
    root.geometry('1000x700')
    root.mainloop()


if __name__ == '__main__':
    main()
