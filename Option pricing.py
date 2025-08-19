#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Option Pricer & Greeks (GUI)
Author: Your Name
License: MIT

A single-file Tkinter GUI for European option pricing using the Black-Scholes-Merton model,
including dividend yield (q), full set of Greeks, and implied volatility solver.

Formulas:
  d1 = [ln(S/K) + (r - q + 0.5*sigma^2)T] / (sigma*sqrt(T))
  d2 = d1 - sigma*sqrt(T)
  Call = S*e^(-qT) N(d1) - K*e^(-rT) N(d2)
  Put  = K*e^(-rT) N(-d2) - S*e^(-qT) N(-d1)

Greeks use standard BSM with continuous dividend yield.
Implied vol solved via adaptive bracketing + bisection (monotonic in sigma).

No third-party dependencies required.
"""

import csv
import math
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

# -----------------------------
# Math utilities (no numpy)
# -----------------------------

SQRT_2PI = math.sqrt(2.0 * math.pi)

def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI

def norm_cdf(x: float) -> float:
    # Using math.erf for cumulative normal
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# -----------------------------
# Black-Scholes functions
# -----------------------------

def _d1_d2(S, K, r, q, sigma, T):
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma, and T must be positive.")
    vsqrtT = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrtT
    d2 = d1 - vsqrtT
    return d1, d2

def bs_price(S, K, r, q, sigma, T, is_call: bool) -> float:
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    if is_call:
        return S * df_q * norm_cdf(d1) - K * df_r * norm_cdf(d2)
    else:
        return K * df_r * norm_cdf(-d2) - S * df_q * norm_cdf(-d1)

def bs_greeks(S, K, r, q, sigma, T, is_call: bool):
    d1, d2 = _d1_d2(S, K, r, q, sigma, T)
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    n_d1 = norm_pdf(d1)

    # Delta
    if is_call:
        delta = df_q * norm_cdf(d1)
    else:
        delta = -df_q * norm_cdf(-d1)

    # Gamma
    gamma = (df_q * n_d1) / (S * sigma * math.sqrt(T))

    # Vega (per 1.00 change in vol, i.e., volatility in decimals)
    vega = S * df_q * n_d1 * math.sqrt(T)

    # Theta (per year)
    term1 = -(S * df_q * n_d1 * sigma) / (2.0 * math.sqrt(T))
    if is_call:
        theta = term1 - r * K * df_r * norm_cdf(d2) + q * S * df_q * norm_cdf(d1)
    else:
        theta = term1 + r * K * df_r * norm_cdf(-d2) - q * S * df_q * norm_cdf(-d1)
    theta_per_day = theta / 365.0  # calendar days

    # Rho
    if is_call:
        rho = K * T * df_r * norm_cdf(d2)
    else:
        rho = -K * T * df_r * norm_cdf(-d2)

    return {
        "Delta": delta,
        "Gamma": gamma,
        "Vega": vega,
        "Theta (per year)": theta,
        "Theta (per day)": theta_per_day,
        "Rho": rho,
    }

# -----------------------------
# Implied volatility solver
# -----------------------------

def implied_vol(
    market_price: float,
    S: float, K: float, r: float, q: float, T: float, is_call: bool,
    tol: float = 1e-7, max_iter: int = 150
) -> float:
    """
    Robust IV via bisection with adaptive bracketing.
    Returns sigma in decimal (e.g., 0.20 for 20%).
    Raises ValueError if not solvable (e.g., price out of no-arbitrage bounds).
    """
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        raise ValueError("Inputs must be positive and market price > 0.")

    # Basic no-arbitrage bounds for European options with yield
    df_r = math.exp(-r * T)
    df_q = math.exp(-q * T)
    intrinsic_call = max(0.0, S * df_q - K * df_r)
    intrinsic_put  = max(0.0, K * df_r - S * df_q)
    upper_bound = S * df_q if is_call else K * df_r  # simple loose upper bounds

    if is_call:
        if not (intrinsic_call <= market_price <= upper_bound + 1e-12):
            raise ValueError("Market price outside reasonable bounds for a call.")
    else:
        if not (intrinsic_put <= market_price <= upper_bound + 1e-12):
            raise ValueError("Market price outside reasonable bounds for a put.")

    # Initial bracket
    lo, hi = 1e-6, 3.0
    price_lo = bs_price(S, K, r, q, lo, T, is_call)
    price_hi = bs_price(S, K, r, q, hi, T, is_call)

    # Expand high if needed (cap to avoid runaway)
    expand_count = 0
    while price_hi < market_price and hi < 5.0 and expand_count < 15:
        hi *= 1.4
        price_hi = bs_price(S, K, r, q, hi, T, is_call)
        expand_count += 1

    if price_lo > market_price or price_hi < market_price:
        # Could not bracket; likely due to numerical/bounds issues
        raise ValueError("Failed to bracket implied volatility for given inputs.")

    # Bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price_mid = bs_price(S, K, r, q, mid, T, is_call)
        if abs(price_mid - market_price) < tol:
            return mid
        if price_mid < market_price:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# -----------------------------
# GUI
# -----------------------------

class OptionPricerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Option Pricer & Greeks (Black–Scholes–Merton)")
        self.geometry("720x530")
        self.minsize(700, 520)

        # Ttk theme
        try:
            self.style = ttk.Style(self)
            if "clam" in self.style.theme_names():
                self.style.theme_use("clam")
        except Exception:
            pass

        self._build_menu()
        self._build_widgets()

    def _build_menu(self):
        menubar = tk.Menu(self)
        # File
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Save Results as CSV…", command=self.save_results_csv)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)

        # Edit
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Copy Results to Clipboard", command=self.copy_results)
        menubar.add_cascade(label="Edit", menu=editmenu)

        # Help
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=helpmenu)

        self.config(menu=menubar)

    def _build_widgets(self):
        pad = {"padx": 10, "pady": 6}

        # Container frames
        input_frame = ttk.LabelFrame(self, text="Inputs")
        input_frame.pack(fill="x", padx=12, pady=10)

        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=12, pady=(0, 10))

        output_frame = ttk.LabelFrame(self, text="Results")
        output_frame.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Inputs
        self.var_is_call = tk.BooleanVar(value=True)
        rb_call = ttk.Radiobutton(input_frame, text="Call", variable=self.var_is_call, value=True)
        rb_put = ttk.Radiobutton(input_frame, text="Put", variable=self.var_is_call, value=False)
        rb_call.grid(row=0, column=0, sticky="w", **pad)
        rb_put.grid(row=0, column=1, sticky="w", **pad)

        self.ent_S = self._labeled_entry(input_frame, "Spot Price (S)", 1, 0, default="100")
        self.ent_K = self._labeled_entry(input_frame, "Strike (K)",    1, 2, default="100")
        self.ent_r = self._labeled_entry(input_frame, "Risk-free r (dec.)", 2, 0, default="0.05")
        self.ent_q = self._labeled_entry(input_frame, "Dividend q (dec.)",  2, 2, default="0.02")
        self.ent_Td= self._labeled_entry(input_frame, "Time to Expiry (days)", 3, 0, default="180")
        self.ent_sigma = self._labeled_entry(input_frame, "Volatility σ (dec.)", 3, 2, default="0.20")
        self.ent_mkt = self._labeled_entry(input_frame, "Market Price (optional)", 4, 0, default="")

        # Stretch columns
        input_frame.grid_columnconfigure(1, weight=1)
        input_frame.grid_columnconfigure(3, weight=1)

        # Actions
        btn_price = ttk.Button(action_frame, text="Compute Price & Greeks", command=self.compute_price_greeks)
        btn_iv    = ttk.Button(action_frame, text="Solve Implied Vol (from Market Price)", command=self.compute_iv)
        btn_clear = ttk.Button(action_frame, text="Clear Results", command=self.clear_results)
        btn_price.pack(side="left", padx=6, pady=6)
        btn_iv.pack(side="left", padx=6, pady=6)
        btn_clear.pack(side="left", padx=6, pady=6)

        # Results tree
        self.tree = ttk.Treeview(output_frame, columns=("metric", "value"), show="headings", height=14)
        self.tree.heading("metric", text="Metric")
        self.tree.heading("value", text="Value")
        self.tree.column("metric", width=240, anchor="w")
        self.tree.column("value", width=200, anchor="e")

        vsb = ttk.Scrollbar(output_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
        vsb.pack(side="left", fill="y", padx=(0, 10), pady=10)

    def _labeled_entry(self, parent, label, row, col, default=""):
        lbl = ttk.Label(parent, text=label)
        lbl.grid(row=row, column=col, sticky="w", padx=10, pady=4)
        ent = ttk.Entry(parent)
        ent.insert(0, default)
        ent.grid(row=row, column=col+1, sticky="ew", padx=10, pady=4)
        return ent

    # -----------------------------
    # Actions
    # -----------------------------

    def _read_inputs(self, need_sigma: bool = True, need_market: bool = False):
        try:
            S = float(self.ent_S.get())
            K = float(self.ent_K.get())
            r = float(self.ent_r.get())
            q = float(self.ent_q.get())
            days = float(self.ent_Td.get())
            if days <= 0:
                raise ValueError("Time to expiry (days) must be positive.")
            T = days / 365.0
            sigma = None
            if need_sigma:
                sigma_str = self.ent_sigma.get().strip()
                sigma = float(sigma_str)
                if sigma <= 0:
                    raise ValueError("Volatility must be positive.")
            mkt = None
            if need_market:
                mkt_str = self.ent_mkt.get().strip()
                if not mkt_str:
                    raise ValueError("Please enter Market Price to solve implied volatility.")
                mkt = float(mkt_str)
                if mkt <= 0:
                    raise ValueError("Market price must be positive.")
            return S, K, r, q, T, sigma, mkt
        except ValueError as e:
            messagebox.showerror("Input Error", f"{e}")
            return None

    def compute_price_greeks(self):
        vals = self._read_inputs(need_sigma=True, need_market=False)
        if not vals:
            return
        S, K, r, q, T, sigma, _ = vals
        is_call = self.var_is_call.get()
        try:
            price = bs_price(S, K, r, q, sigma, T, is_call)
            gr = bs_greeks(S, K, r, q, sigma, T, is_call)
        except ValueError as e:
            messagebox.showerror("Computation Error", str(e))
            return

        self.clear_results()
        self._insert_result("Option Type", "Call" if is_call else "Put")
        self._insert_result("Spot (S)", f"{S:,.4f}")
        self._insert_result("Strike (K)", f"{K:,.4f}")
        self._insert_result("r (annual, dec.)", f"{r:.6f}")
        self._insert_result("q (annual, dec.)", f"{q:.6f}")
        self._insert_result("T (years)", f"{T:.6f}")
        self._insert_result("σ (vol, dec.)", f"{sigma:.6f}")
        self._insert_result("Price", f"{price:,.6f}")
        for k, v in gr.items():
            self._insert_result(k, f"{v:,.6f}")

    def compute_iv(self):
        vals = self._read_inputs(need_sigma=False, need_market=True)
        if not vals:
            return
        S, K, r, q, T, _, mkt = vals
        is_call = self.var_is_call.get()
        try:
            iv = implied_vol(mkt, S, K, r, q, T, is_call)
        except ValueError as e:
            messagebox.showerror("IV Solver", str(e))
            return

        # Also compute Greeks at the solved IV
        price = bs_price(S, K, r, q, iv, T, is_call)
        gr = bs_greeks(S, K, r, q, iv, T, is_call)

        self.clear_results()
        self._insert_result("Option Type", "Call" if is_call else "Put")
        self._insert_result("Market Price", f"{mkt:,.6f}")
        self._insert_result("Solved σ (dec.)", f"{iv:.6f}")
        self._insert_result("Implied Vol (%)", f"{iv*100:.4f}%")
        self._insert_result("Model Price @ σ*", f"{price:,.6f}")
        for k, v in gr.items():
            self._insert_result(k, f"{v:,.6f}")

    def _insert_result(self, metric, value):
        self.tree.insert("", "end", values=(metric, value))

    def clear_results(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

    def copy_results(self):
        rows = [self.tree.item(i, "values") for i in self.tree.get_children()]
        if not rows:
            messagebox.showinfo("Copy", "No results to copy yet.")
            return
        text = "\n".join(f"{m}: {v}" for m, v in rows)
        self.clipboard_clear()
        self.clipboard_append(text)
        messagebox.showinfo("Copy", "Results copied to clipboard.")

    def save_results_csv(self):
        rows = [self.tree.item(i, "values") for i in self.tree.get_children()]
        if not rows:
            messagebox.showinfo("Save CSV", "No results to save yet.")
            return
        fpath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if not fpath:
            return
        try:
            with open(fpath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for m, v in rows:
                    writer.writerow([m, v])
            messagebox.showinfo("Save CSV", f"Saved to:\n{fpath}")
        except Exception as e:
            messagebox.showerror("Save CSV", f"Failed to save:\n{e}")

    def show_about(self):
        message = (
            "Option Pricer & Greeks (Black–Scholes–Merton)\n"
            "• European options with dividend yield\n"
            "• Price, full Greeks, and Implied Volatility\n\n"
            "No external dependencies · MIT License"
        )
        messagebox.showinfo("About", message)

# -----------------------------
# Entrypoint
# -----------------------------

def main():
    app = OptionPricerApp()
    # HiDPI scaling hint (Windows); harmless elsewhere
    try:
        if sys.platform.startswith("win"):
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    app.mainloop()

if __name__ == "__main__":
    main()
