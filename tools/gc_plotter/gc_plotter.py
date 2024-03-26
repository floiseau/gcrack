import tkinter as tk
from tkinter import ttk

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Configure Matplotlib to use LaTeX font
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
    }
)


def plot_polar_function():
    gamma = sp.symbols("gamma")
    try:
        # Get the text from the entire entry widget
        func_text = entry.get()
        # Convert it into an expression
        gamma_expr = sp.sympify(
            func_text, locals={"gamma": gamma}
        )  # Convert text to SymPy expression
        # Generate a Python function for the expression
        gamma_f = sp.lambdify([gamma], gamma_expr)
        # Apply the function
        gamma_vals = np.linspace(0, 2 * np.pi, 1000)
        gc_vals = gamma_f(gamma_vals)
        # Update the plot
        ax.clear()
        ax.plot(gamma_vals, gc_vals)
        ax.set_title(
            r"$G_c (\gamma) =" + sp.latex(gamma_expr) + "$"
        )  # Convert expression to LaTeX
        ax.grid(True)
        canvas.draw()
    except Exception as e:
        print("Error:", e)


def export_plot_as_pdf():
    file_path = tk.filedialog.asksaveasfilename(defaultextension=".pdf")
    if file_path:
        canvas.print_figure(file_path, format="pdf")


# Create main window
root = tk.Tk()
root.title("Polar Plotter")

# Create a frame for the title and description
title_frame = ttk.Frame(root)
title_frame.pack(padx=10, pady=10)

# Add a title
title_label = ttk.Label(
    title_frame, text="Gc Plotter", font=("Helvetica", 20, "bold"), foreground="blue"
)
title_label.pack()

# Add a description to explain how to use the program
description_label = ttk.Label(
    title_frame,
    text="Enter a function of gamma in the text box below and press 'Plot'.\n"
    "You can also export the plot as a PDF using the 'Export PDF' button.",
    font=("Helvetica", 12),
)
description_label.pack()

# Create a frame for the plot
plot_frame = tk.Frame(root)
plot_frame.pack(padx=10, pady=10)

# Create a canvas for the plot
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, polar=True)
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Create a frame for the text box and button
input_frame = tk.Frame(root)
input_frame.pack(padx=10, pady=10)

# Create a text box for entering the function
entry = tk.Entry(input_frame, width=30, font=("Helvetica", 12))
entry.pack(side=tk.LEFT, padx=5, pady=5)

# Create a button to plot the function
plot_button = tk.Button(
    input_frame,
    text="Plot",
    command=plot_polar_function,
    font=("Helvetica", 12, "bold"),
    foreground="green",
    padx=10,
    pady=5,
)
plot_button.pack(side=tk.LEFT, padx=5, pady=5)

# Create a button to export the plot as a PDF
export_button = tk.Button(
    input_frame,
    text="Export PDF",
    command=export_plot_as_pdf,
    font=("Helvetica", 12, "bold"),
    foreground="red",
    padx=10,
    pady=5,
)
export_button.pack(side=tk.LEFT, padx=5, pady=5)

# Run the Tkinter event loop
root.mainloop()
