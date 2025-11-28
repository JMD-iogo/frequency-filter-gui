""" 
Autor: Diogo Ferreira
"""

# NAME:
#   frequency_filter_gui.py
#
# PURPOSE:
#   Aplicação interativa em Python/Tkinter para aplicar filtros no domínio da
#   frequência (LPF, HPF, BPF e BRF/rejeita-banda) a imagens 2D (tons de
#   cinzento ou RGB), visualizar espectros de magnitude/fase e avaliar a
#   qualidade da filtragem com MSE e PSNR.
#
# CATEGORY:
#   Processamento Digital de Imagem / Processamento de Sinal no domínio da frequência.
#
# CALLING SEQUENCE:
#   - A partir da linha de comandos: python frequency_filter_gui.py  ou
#   - Correndo o código diretamente no ambiente de desenvolvimento  ou
#   - Abrindo o executável gerado a partir deste script.
#
# INPUTS:
#   Entrada principal da aplicação (via GUI):
#     - Ficheiro de imagem selecionado pelo utilizador
#       (formatos suportados: PNG, JPG/JPEG, BMP, TIFF, etc.).
#
# GUI PARAMETERS:
#   (Selecionados/ajustados via interface gráfica)
#
#   Tipo de filtro (filter_type_var / OptionMenu):
#       'LPF (passa-baixo)'  – filtro passa-baixo radial centrado.
#       'HPF (passa-alto)'   – filtro passa-alto radial centrado.
#       'BPF (passa-banda)'  – filtro passa-banda radial centrado.
#       'BRF (rejeita banda)'– filtro rejeita-banda radial centrado.
#
#
#   Perfil do filtro (profile_var / OptionMenu):
#       'Ideal'       – máscara binária ideal.
#       'Gaussiano'   – máscara gaussiana.
#       'Butterworth' – máscara Butterworth de ordem n.
#
#   Parâmetros radiais (sliders r1, r2, ordem):
#       r1  – raio interior em píxeis
#             (usado em HPF, BPF e BRF).
#       r2  – raio exterior em píxeis
#             (usado em LPF, BPF e BRF).
#       n   – ordem Butterworth (scale_order), aplicável quando o perfil é
#             'Butterworth' (quanto maior n, mais abrupta a transição).
#
#   Parâmetros de filtro deslocado / notch:
#       Checkbox "Filtro deslocado (tipo notch)":
#         - Quando desligada: o filtro é centrado na frequência 0 (centro da FFT).
#         - Quando ligada: o centro do filtro é deslocado para (u0, v0), permitindo
#           criar filtros tipo notch (rejeição localizada em frequência).
#
#       u0, v0 – deslocamento do centro do filtro relativamente ao centro da
#                transformada (em píxeis, eixo das colunas/linhas).
#                Podem ser definidos:
#                  · manualmente, através das escalas (scale_notch_u0, scale_notch_v0)
#                  · ou clicando diretamente em algumas das imagens da figura
#                    principal (original, magnitude, fase, magnitude filtrada,
#                    imagem filtrada).
#
#   Definições adicionais (janela "Definições"):
#       - "Atualizar máscara em tempo real":
#           Atualiza o preview da máscara automaticamente ao alterar
#           parâmetros (r1, r2, ordem, u0, v0, tipo/perfil).
#       - "Aplicar filtro em tempo real" :
#           Aplica o filtro imediatamente após qualquer alteração de parâmetros.
#       - "Mostrar dicas (tooltips)":
#           Liga/desliga globalmente as dicas, estas aparecem ao passar o rato sobre
#           os widgets da GUI.
#
#
# OUTPUTS:
#   Resultados mostrados na figura principal:
#       - Imagem original (domínio espacial).
#       - Magnitude do espectro original (escala logarítmica normalizada).
#       - Fase do espectro original.
#       - Máscara do filtro no domínio da frequência.
#       - Magnitude do espectro filtrado.
#       - Imagem filtrada reconstruída no domínio espacial.
#
#   Métricas numéricas (label na GUI, painel direito):
#       - MSE  – erro quadrático médio entre imagem original e filtrada.
#       - PSNR – relação sinal-ruído de pico (em dB), assumindo imagens
#                normalizadas em [0, 1].
#
#   Output opcional em ficheiro (botão "Guardar imagem filtrada"):
#       - Imagem filtrada, no formato selecionado (PNG/JPEG/BMP/TIFF).
#       - Opcionalmente, caso as checkboxes estejam ativas, é criada uma pasta
#         com o nome base do ficheiro escolhido, contendo:
#           * Imagem original.
#           * Magnitude do espectro original.
#           * Fase do espectro original.
#           * Máscara do filtro.
#           * Magnitude do espectro filtrado.
#           * Ficheiro de texto "<base>_spcs.txt" com:
#               · caminho da imagem filtrada,
#               · tipo de filtro (LPF/HPF/BPF/BRF) e perfil (Ideal/Gaussiano/Butterworth),
#               · parâmetros (r1, r2, ordem, u0, v0, estado do filtro deslocado),
#               · MSE e PSNR.
#
#
# SIDE EFFECTS:
#   - Criação de janelas gráficas Tkinter (janela principal e janela de
#     definições).
#   - Criação de subdiretório para saída, quando o utilizador seleciona
#     guardar ficheiros adicionais (imagem original, espectros, máscara, specs).
#   - Possível aumento do tempo de resposta quando a atualização em tempo real
#     da máscara e/ou aplicação automática do filtro estão ligadas, sobretudo
#     para imagens grandes.
#
#
# RESTRICTIONS:
#   - Tamanho máximo da imagem carregada limitado por 'max_side' em
#     load_image_any (por defeito 512 píxeis no maior lado).
#   - A aplicação assume imagens normalizadas em [0, 1] para o cálculo de MSE
#     e PSNR.
#   - Em imagens RGB, o utilizador escolhe:
#       -converter para intensidade (tons de cinzento) ou
#       -aplicar o filtro de igual fomra em cada canal
#
#
# PROCEDURE:
#   1. Carregamento da imagem.
#      Se a imagem for RGB, o utilizador escolhe trabalhar em intensidade
#      (conversão para tons de cinzento) ou aplicar o filtro canal-a-canal.
#   2. Cálculo da Transformada de Fourier 2D, seguida de fftshift
#      para centrar as frequências baixas.
#   3. Cálculo do mapa de distâncias radiais D em relação ao centro da FFT.
#   4. Criação da máscara de filtro no domínio da frequência de acordo com as seleções
#   5. Aplicação da máscara sobre o espectro (canal a canal para RGB).
#   6. Transformada inversa (ifftshift + ifft2) e extração da parte real.
#   7. Atualização das figuras (original, espectros, máscara, filtrada) e
#      cálculo das métricas MSE e PSNR.
#   8. Opcionalmente, gravação dos resultados em disco (imagem filtrada,
#      componentes auxiliares e ficheiro de especificações).
#
#

import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from typing import Literal

# ----------------------------------------------------------------------
# Funções utilitárias
# ----------------------------------------------------------------------


def load_image_any(path: str, max_side: int = 512) -> tuple[np.ndarray, Literal["gray", "rgb"]]:
    img = Image.open(path)

    # Redimensionar se necessário
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.BICUBIC)

    # Ver se é intensidade ou cor
    if img.mode in ("L", "I;16", "I", "F"):
        # Já é intensidade
        img_g = img.convert("L")
        arr = np.array(img_g, dtype=np.float32) / 255.0
        return arr, "gray"
    else:
        # Consideramos que é cor → convertemos para RGB
        img_rgb = img.convert("RGB")
        arr = np.array(img_rgb, dtype=np.float32) / 255.0
        return arr, "rgb"


def precompute_distance_map(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X**2 + Y**2)
    return D


def compute_magnitude(F_shift: np.ndarray) -> np.ndarray:
    """ Se F_shift for 3D (H,W,3), calcula magnitude escalar por pixel (norma). """
    if F_shift.ndim == 3:
        mag = np.sqrt(np.sum(np.abs(F_shift) ** 2, axis=-1))
    else:
        mag = np.abs(F_shift)
    magnitude = np.log1p(mag)
    mmax = magnitude.max()
    if mmax > 0:
        magnitude = magnitude / mmax
    return magnitude


def compute_phase(F_shift: np.ndarray) -> np.ndarray:
    """ Se F_shift for 3D, usa a fase da primeira componente como representação. """
    if F_shift.ndim == 3:
        phase = np.angle(F_shift[..., 0])
    else:
        phase = np.angle(F_shift)
    phase = (phase + np.pi) / (2 * np.pi)
    return phase


def create_frequency_filter(
    D: np.ndarray,
    filter_type: str,
    profile_type: str,
    r1: float,
    r2: float,
    order: int,
    center_offset: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Cria máscara de filtro no domínio da frequência.

    filter_type: 'LPF', 'HPF', 'BPF' (passa-banda), 'BRF' (rejeita banda).
    profile_type: 'Ideal', 'Gaussiano', 'Butterworth'.
    center_offset:
      - None → filtro centrado em (0,0) (frequência baixa no centro da FFT).
      - (u0,v0) → filtro deslocado (tipo notch), centro relativo ao centro da FFT.
    """
    if D is None:
        raise ValueError("Mapa de distâncias D não pode ser None.")

    filter_type = filter_type.upper()
    profile_type = profile_type.capitalize()

    rows, cols = D.shape
    crow, ccol = rows // 2, cols // 2

    # Mapa de distâncias em relação ao centro escolhido
    if center_offset is None:
        D0 = D
    else:
        u0, v0 = center_offset
        y = np.arange(rows)
        x = np.arange(cols)
        X, Y = np.meshgrid(x, y)
        u_c = ccol + u0
        v_c = crow + v0
        D0 = np.sqrt((X - u_c) ** 2 + (Y - v_c) ** 2)

    if filter_type not in ("LPF", "HPF", "BPF", "BRF"):
        raise ValueError("Tipo de filtro inválido.")
    if profile_type not in ("Ideal", "Gaussiano", "Butterworth"):
        raise ValueError("Tipo de perfil de filtro inválido.")

    mask = np.zeros_like(D0, dtype=np.float32)

    # ------------------------------
    # Perfil Ideal
    # ------------------------------
    if profile_type == "Ideal":
        if filter_type == "LPF":
            mask[D0 <= r2] = 1.0
        elif filter_type == "HPF":
            mask[D0 >= r1] = 1.0
        elif filter_type == "BPF":
            mask[(D0 >= r1) & (D0 <= r2)] = 1.0
        elif filter_type == "BRF":
            mask[:, :] = 1.0
            mask[(D0 >= r1) & (D0 <= r2)] = 0.0
        return mask

    # ------------------------------
    # Perfil Gaussiano
    # ------------------------------
    if profile_type == "Gaussiano":
        if filter_type == "LPF":
            D0c = max(float(r2), 1e-6)
            mask = np.exp(-(D0**2) / (2 * D0c**2))
        elif filter_type == "HPF":
            D0c = max(float(r1), 1e-6)
            mask = 1.0 - np.exp(-(D0**2) / (2 * D0c**2))
        elif filter_type in ("BPF", "BRF"):
            D0_low = max(float(r1), 1e-6)
            D0_high = max(float(r2), D0_low + 1e-6)

            lpf_high = np.exp(-(D0**2) / (2 * D0_high**2))
            lpf_low = np.exp(-(D0**2) / (2 * D0_low**2))
            bpf = lpf_high - lpf_low
            bpf = np.clip(bpf, 0.0, 1.0)

            if filter_type == "BPF":
                mask = bpf
            else:  # BRF
                mask = 1.0 - bpf
        return mask

    # ------------------------------
    # Perfil Butterworth
    # ------------------------------
    if profile_type == "Butterworth":
        n = max(int(order), 1)

        def butterworth_lpf(D0c: float) -> np.ndarray:
            D0c = max(float(D0c), 1e-6)
            return 1.0 / (1.0 + (D0 / D0c) ** (2 * n))

        if filter_type == "LPF":
            mask = butterworth_lpf(r2)
        elif filter_type == "HPF":
            mask = 1.0 - butterworth_lpf(r1)
        elif filter_type in ("BPF", "BRF"):
            D0_low = max(float(r1), 1e-6)
            D0_high = max(float(r2), D0_low + 1e-6)
            lpf_high = butterworth_lpf(D0_high)
            lpf_low = butterworth_lpf(D0_low)
            bpf = lpf_high - lpf_low
            bpf = np.clip(bpf, 0.0, 1.0)

            if filter_type == "BPF":
                mask = bpf
            else:  # BRF
                mask = 1.0 - bpf
        return mask

    raise ValueError("Tipo de perfil de filtro inválido.")


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(np.mean((img1 - img2) ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    m = mse(img1, img2)
    if m == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / m)


def save_array_as_png(arr: np.ndarray, path: str) -> None:
    """Normaliza array [min,max] para [0,255] e guarda como PNG."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = arr - arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr = arr / max_val
    img = (arr * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# ----------------------------------------------------------------------
# Classe de Tooltip simples (com controlo global)
# ----------------------------------------------------------------------


class ToolTip:
    """
    Tooltip simples que aparece quando o rato passa por cima de um widget.
      nota - ToolTip.enabled = True/False para ligar/desligar todas as dicas.
    """

    enabled: bool = True
    instances: list["ToolTip"] = []

    def __init__(self, widget: tk.Widget, text: str, wait_ms: int = 500) -> None:
        self.widget = widget
        self.text = text
        self.wait_ms = wait_ms
        self.tipwindow: tk.Toplevel | None = None
        self.id_after: str | None = None

        ToolTip.instances.append(self)

        widget.bind("<Enter>", self.on_enter)
        widget.bind("<Leave>", self.on_leave)

    def on_enter(self, _event=None) -> None:
        if not ToolTip.enabled:
            return
        self.schedule()

    def on_leave(self, _event=None) -> None:
        self.unschedule()
        self.hidetip()

    def schedule(self) -> None:
        self.unschedule()
        self.id_after = self.widget.after(self.wait_ms, self.showtip)

    def unschedule(self) -> None:
        if self.id_after is not None:
            self.widget.after_cancel(self.id_after)
            self.id_after = None

    def showtip(self) -> None:
        if not ToolTip.enabled:
            return
        if self.tipwindow is not None:
            return
        if self.widget.winfo_ismapped():
            x, y, _cx, cy = self.widget.bbox("insert")
        else:
            x, y, cy = 0, 0, 0
        x = x + self.widget.winfo_rootx() + 5
        y = y + self.widget.winfo_rooty() + cy - 20

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            padx=4,
            pady=2,
            font=("Segoe UI", 8),
        )
        label.pack(ipadx=1)

    def hidetip(self) -> None:
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


# ----------------------------------------------------------------------
# Classe GUI
# ----------------------------------------------------------------------


class FrequencyFilterGUI:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        master.title("Filtros de Frequência (LPF / HPF / BPF / BRF)")
        master.resizable(True, True)
        master.minsize(900, 850)

        #Para contraste visual
        bg_main = "#f2f2f7"
        bg_panel = "#e6e6f2"
        master.configure(bg=bg_main)

        # Estado interno
        self.img_original: np.ndarray | None = None  # (H,W) ou (H,W,3)
        self.img_filtered: np.ndarray | None = None  # (H,W) ou (H,W,3)
        self.D: np.ndarray | None = None
        self.F_shift: np.ndarray | None = None  # fftshift
        self.magnitude_original: np.ndarray | None = None
        self.magnitude_filtered: np.ndarray | None = None
        self.phase_original: np.ndarray | None = None
        self.filter_mask: np.ndarray | None = None
        self.img_path: str | None = None

        # Estado das dicas / tooltips
        self.tips_var = tk.BooleanVar(value=True)
        ToolTip.enabled = True

        # Estado da janela de definições
        self.settings_window: tk.Toplevel | None = None

        # Estado de tempo real
        self.realtime_mask_var = tk.BooleanVar(value=False)
        self.realtime_mask_warned = False
        self.auto_apply_var = tk.BooleanVar(value=False)
        self.auto_apply_warned = False

        # Centro deslocado (notch)
        self.center_shift_var = tk.BooleanVar(value=False)
        self.center_shift_warned = False

        # ------------------------------------------------------------------
        # Painéis principais
        # ------------------------------------------------------------------
        control_frame = tk.LabelFrame(
            master,
            text="Controlo",
            padx=5,
            pady=5,
            bg=bg_panel,
            font=("Segoe UI", 9, "bold"),
            bd=2,
            relief=tk.GROOVE,
        )
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 3), pady=5)

        right_panel = tk.LabelFrame(
            master,
            text="Resultados e saída",
            padx=5,
            pady=5,
            bg=bg_panel,
            font=("Segoe UI", 9, "bold"),
            bd=2,
            relief=tk.GROOVE,
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 5), pady=5)

        figure_frame = tk.Frame(master, bg=bg_main)
        figure_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=5)

        # ================= PAINEL ESQUERDO ================================
        # === Imagem =======================================================
        image_frame = tk.LabelFrame(control_frame, text="Imagem", padx=5, pady=5, bg=bg_panel)
        image_frame.pack(fill=tk.X, pady=(0, 8))

        btn_load = tk.Button(
            image_frame,
            text="Carregar imagem",
            command=self.load_image_callback,
            font=("Segoe UI", 9, "bold"),
            bg="#dde5ff",
            activebackground="#c7d5ff",
        )
        btn_load.pack(fill=tk.X, pady=2)
        ToolTip(btn_load, "Escolhe uma imagem (PNG, JPG, BMP, TIFF) para filtrar.")

        # Botão Definições logo por baixo de "Carregar imagem"
        btn_settings = tk.Button(
            image_frame,
            text="Abrir definições",
            command=self.open_settings_window,
            font=("Segoe UI", 9),
            bg="#e0e0ff",
            activebackground="#c7c7ff",
        )
        btn_settings.pack(fill=tk.X, pady=(4, 0))
        ToolTip(btn_settings, "Abre as definições: tempo real e dicas (tooltips).")

        # === Tipo de filtro + perfil ======================================
        filter_frame = tk.LabelFrame(
            control_frame, text="Filtro (tipo e perfil)", padx=5, pady=5, bg=bg_panel
        )
        filter_frame.pack(fill=tk.X, pady=(0, 8))

        # Tipo de filtro (OptionMenu com texto completo)
        lbl_filter = tk.Label(filter_frame, text="Tipo de filtro:", bg=bg_panel)
        lbl_filter.pack(anchor="w")

        self.filter_type_var = tk.StringVar(value="LPF (passa-baixo)")
        filter_type_options = [
            "LPF (passa-baixo)",
            "HPF (passa-alto)",
            "BPF (passa-banda)",
            "BRF (rejeita banda)",
        ]
        self.filter_menu = tk.OptionMenu(filter_frame, self.filter_type_var, *filter_type_options)
        self.filter_menu.config(font=("Segoe UI", 9))
        self.filter_menu.pack(fill=tk.X, pady=(0, 3))
        ToolTip(
            self.filter_menu,
            "Escolhe o tipo de filtro: passa-baixo, passa-alto, passa-banda ou rejeita-banda.",
        )

        # Perfil
        lbl_profile = tk.Label(filter_frame, text="Perfil de filtro:", bg=bg_panel)
        lbl_profile.pack(anchor="w", pady=(4, 0))

        self.profile_var = tk.StringVar(value="Ideal")
        profile_options = ["Ideal", "Gaussiano", "Butterworth"]
        self.profile_menu = tk.OptionMenu(filter_frame, self.profile_var, *profile_options)
        self.profile_menu.config(font=("Segoe UI", 9))
        self.profile_menu.pack(fill=tk.X)
        ToolTip(
            self.profile_menu,
            "Perfil do filtro: Ideal (corte abrupto), Gaussiano (suave) ou Butterworth (controlado pela ordem).",
        )

        # Checkbox "Filtro deslocado (tipo notch)" neste bloco
        self.chk_center_shift = tk.Checkbutton(
            filter_frame,
            text="Filtro deslocado (tipo notch)",
            variable=self.center_shift_var,
            command=self.on_center_shift_toggle,
            bg=bg_panel,
        )
        self.chk_center_shift.pack(fill=tk.X, pady=(5, 0))
        ToolTip(
            self.chk_center_shift,
            "Ativa o modo notch: o filtro é centrado em (u0, v0) em vez de 0. "
            "Podes escolher o centro clicando nas imagens.",
        )

        # === Parâmetros (r1, r2, ordem, notch, u0, v0) ====================
        self.params_frame = tk.LabelFrame(
            control_frame, text="Parâmetros do filtro", padx=5, pady=5, bg=bg_panel
        )
        self.params_frame.pack(fill=tk.X, pady=(0, 8))

        # r1
        self.r1_frame = tk.Frame(self.params_frame, bg=bg_panel)
        self.lbl_r1 = tk.Label(self.r1_frame, text="Raio interior r1 [pixels]:", bg=bg_panel)
        self.lbl_r1.pack(anchor="w")
        self.scale_r1 = tk.Scale(
            self.r1_frame, from_=0, to=200, orient=tk.HORIZONTAL, showvalue=False
        )
        self.scale_r1.set(20)
        self.scale_r1.pack(fill=tk.X)
        self.lbl_r1_val = tk.Label(self.r1_frame, text=f"{self.scale_r1.get()} px", bg=bg_panel)
        self.lbl_r1_val.pack(anchor="e")
        ToolTip(
            self.r1_frame,
            "Raio interior (r1) em pixels. Utilizado em filtros passa-alto, passa-banda e rejeita-banda.",
        )

        # r2
        self.r2_frame = tk.Frame(self.params_frame, bg=bg_panel)
        self.lbl_r2 = tk.Label(self.r2_frame, text="Raio exterior r2 [pixels]:", bg=bg_panel)
        self.lbl_r2.pack(anchor="w")
        self.scale_r2 = tk.Scale(
            self.r2_frame, from_=10, to=300, orient=tk.HORIZONTAL, showvalue=False
        )
        self.scale_r2.set(60)
        self.scale_r2.pack(fill=tk.X)
        self.lbl_r2_val = tk.Label(self.r2_frame, text=f"{self.scale_r2.get()} px", bg=bg_panel)
        self.lbl_r2_val.pack(anchor="e")
        ToolTip(
            self.r2_frame,
            "Raio exterior (r2) em pixels. Define a frequência de corte (LPF) ou limite superior de banda (BPF/BRF).",
        )

        # ordem (Butterworth)
        self.order_frame = tk.Frame(self.params_frame, bg=bg_panel)
        self.lbl_order = tk.Label(self.order_frame, text="Ordem (Butterworth):", bg=bg_panel)
        self.lbl_order.pack(anchor="w")
        self.scale_order = tk.Scale(
            self.order_frame, from_=1, to=10, orient=tk.HORIZONTAL, showvalue=False
        )
        self.scale_order.set(2)
        self.scale_order.pack(fill=tk.X)
        self.lbl_order_val = tk.Label(self.order_frame, text=f"{self.scale_order.get()}", bg=bg_panel)
        self.lbl_order_val.pack(anchor="e")
        ToolTip(
            self.order_frame,
            "Ordem do filtro de Butterworth. Ordens maiores → transição mais abrupta.",
        )

        # Parâmetros de centro (u0,v0) – usados quando o filtro é deslocado
        self.notch_frame = tk.LabelFrame(
            self.params_frame, text="Centro (deslocamento relativo)", bg=bg_panel
        )
        tk.Label(
            self.notch_frame,
            text="u0 (colunas, relativo ao centro):",
            bg=bg_panel,
        ).pack(anchor="w")
        self.scale_notch_u0 = tk.Scale(
            self.notch_frame, from_=-50, to=50, orient=tk.HORIZONTAL, showvalue=False
        )
        self.scale_notch_u0.set(0)
        self.scale_notch_u0.pack(fill=tk.X)
        self.lbl_u0_val = tk.Label(
            self.notch_frame,
            text=f"{self.scale_notch_u0.get()} px",
            bg=bg_panel,
        )
        self.lbl_u0_val.pack(anchor="e")

        tk.Label(
            self.notch_frame,
            text="v0 (linhas, relativo ao centro):",
            bg=bg_panel,
        ).pack(anchor="w")
        self.scale_notch_v0 = tk.Scale(
            self.notch_frame, from_=-50, to=50, orient=tk.HORIZONTAL, showvalue=False
        )
        self.scale_notch_v0.set(0)
        self.scale_notch_v0.pack(fill=tk.X)
        self.lbl_v0_val = tk.Label(
            self.notch_frame,
            text=f"{self.scale_notch_v0.get()} px",
            bg=bg_panel,
        )
        self.lbl_v0_val.pack(anchor="e")
        ToolTip(
            self.notch_frame,
            "Coordenadas (u0, v0) do centro notch relativamente ao centro do espectro.",
        )

        # === Atualização/aplicação (apenas botão aplicar) ===============
        exec_frame = tk.LabelFrame(
            control_frame,
            text="Atualização e aplicação",
            padx=5,
            pady=5,
            bg=bg_panel,
        )
        exec_frame.pack(fill=tk.X, pady=(0, 8))

        btn_apply = tk.Button(
            exec_frame,
            text="Aplicar filtro",
            command=self.apply_filter_callback,
            font=("Segoe UI", 9, "bold"),
            bg="#dde5ff",
            activebackground="#c7d5ff",
        )
        btn_apply.pack(fill=tk.X, pady=(2, 0))
        ToolTip(btn_apply, "Aplica o filtro com os parâmetros atuais.")

        # ================= CENTRO: 6 IMAGENS (MATPLOTLIB) ==================
        title_label = tk.Label(
            figure_frame,
            text="Filtragem no domínio da frequência",
            font=("Segoe UI", 11, "bold"),
            bg=bg_main,
        )
        title_label.pack(side=tk.TOP, pady=(0, 4))

        self.fig = Figure(figsize=(9, 6))
        gs = self.fig.add_gridspec(2, 1, height_ratios=[1, 1])
        gs_top = gs[0].subgridspec(1, 3)
        gs_bot = gs[1].subgridspec(1, 3)

        self.ax1 = self.fig.add_subplot(gs_top[0, 0])
        self.ax2 = self.fig.add_subplot(gs_top[0, 1])
        self.ax3 = self.fig.add_subplot(gs_top[0, 2])
        self.ax4 = self.fig.add_subplot(gs_bot[0, 0])
        self.ax5 = self.fig.add_subplot(gs_bot[0, 1])
        self.ax6 = self.fig.add_subplot(gs_bot[0, 2])

        self.canvas = FigureCanvasTkAgg(self.fig, master=figure_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_figure_click)

        self.clear_figure()

        self.fig.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.95,
            bottom=0.05,
            wspace=0.12,
            hspace=0.12,
        )
        self.canvas.draw()

        # ================= PAINEL DIREITO ==================================
        save_frame = tk.LabelFrame(
            right_panel, text="Guardar resultados", padx=5, pady=5, bg=bg_panel
        )
        save_frame.pack(fill=tk.X, pady=(0, 8))

        btn_save = tk.Button(
            save_frame,
            text="Guardar imagem filtrada",
            command=self.save_image_callback,
            font=("Segoe UI", 9, "bold"),
            bg="#dde5ff",
            activebackground="#c7d5ff",
        )
        btn_save.pack(fill=tk.X, pady=(0, 5))
        ToolTip(btn_save, "Guarda a imagem filtrada \n (e outros ficheiros opcionais).")

        extra_frame = tk.LabelFrame(save_frame, text="Guardar também:", bg=bg_panel)
        extra_frame.pack(fill=tk.X, pady=(2, 0))

        self.save_orig_var = tk.BooleanVar(value=False)
        self.save_mag_orig_var = tk.BooleanVar(value=False)
        self.save_phase_var = tk.BooleanVar(value=False)
        self.save_mask_var = tk.BooleanVar(value=False)
        self.save_mag_filt_var = tk.BooleanVar(value=False)
        self.save_specs_var = tk.BooleanVar(value=True)

        tk.Checkbutton(extra_frame, text="Original", variable=self.save_orig_var, bg=bg_panel).pack(
            anchor="w"
        )
        tk.Checkbutton(
            extra_frame,
            text="Magnitude orig.",
            variable=self.save_mag_orig_var,
            bg=bg_panel,
        ).pack(anchor="w")
        tk.Checkbutton(
            extra_frame, text="Fase", variable=self.save_phase_var, bg=bg_panel
        ).pack(anchor="w")
        tk.Checkbutton(
            extra_frame, text="Máscara", variable=self.save_mask_var, bg=bg_panel
        ).pack(anchor="w")
        tk.Checkbutton(
            extra_frame,
            text="Magnitude filtr.",
            variable=self.save_mag_filt_var,
            bg=bg_panel,
        ).pack(anchor="w")
        tk.Checkbutton(
            extra_frame,
            text="Ficheiro spcs",
            variable=self.save_specs_var,
            bg=bg_panel,
        ).pack(anchor="w")

        bottom_frame = tk.LabelFrame(
            right_panel, text="Métricas e notas", padx=5, pady=5, bg=bg_panel
        )
        bottom_frame.pack(fill=tk.X, pady=(0, 0), anchor="n")

        self.metrics_label = tk.Label(
            bottom_frame,
            text="MSE: --\nPSNR: -- dB",
            justify="left",
            bg=bg_panel,
            font=("Segoe UI", 9),
        )
        self.metrics_label.pack(anchor="w", pady=(0, 5))

        info_text = (
            "Notas:\n"
            "MSE – Mean Squared Error \n"
            "PSNR – Peak Signal-to-Noise Ratio"
        
        )
        tk.Label(bottom_frame, text=info_text, justify="left", bg=bg_panel).pack(anchor="w")

        # ------------------------------------------------------------------
        # Ligações de callbacks nos sliders e menus
        # ------------------------------------------------------------------
        self.scale_r1.config(command=self.on_r1_change)
        self.scale_r2.config(command=self.on_r2_change)
        self.scale_order.config(command=self.on_order_change)
        self.scale_notch_u0.config(command=self.on_notch_u0_change)
        self.scale_notch_v0.config(command=self.on_notch_v0_change)

        def on_filter_or_profile_change(*args):
            self.update_controls_visibility()
            if self.realtime_mask_var.get():
                self.update_mask_preview()
            self.maybe_auto_apply()

        self.filter_type_var.trace_add("write", on_filter_or_profile_change)
        self.profile_var.trace_add("write", on_filter_or_profile_change)

        self.update_controls_visibility()

    # ------------------------------------------------------------------
    # Definições / janela de settings
    # ------------------------------------------------------------------

    def open_settings_window(self) -> None:
        """Abre a janela de definições."""
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return

        self.settings_window = tk.Toplevel(self.master)
        self.settings_window.title("Definições")
        self.settings_window.resizable(False, False)
        self.settings_window.transient(self.master)

        bg = "#f7f7ff"
        self.settings_window.configure(bg=bg)

        # Opções de atualização em tempo real
        frame_realtime = tk.LabelFrame(
            self.settings_window,
            text="Atualização em tempo real \n(sem clicar no \"Aplicar Filtro\")",
            padx=10,
            pady=10,
            bg=bg,
        )
        frame_realtime.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))

        chk_realtime_mask = tk.Checkbutton(
            frame_realtime,
            text="Atualizar máscara em tempo real",
            variable=self.realtime_mask_var,
            command=self.on_realtime_mask_toggle,
            bg=bg,
        )
        chk_realtime_mask.pack(anchor="w")

        chk_auto_apply = tk.Checkbutton(
            frame_realtime,
            text="Aplicar filtro em tempo real",
            variable=self.auto_apply_var,
            command=self.on_auto_apply_toggle,
            bg=bg,
        )
        chk_auto_apply.pack(anchor="w", pady=(4, 0))

        # Opções de dicas / tooltips
        frame_tips = tk.LabelFrame(
            self.settings_window,
            text="Ajuda e dicas",
            padx=10,
            pady=10,
            bg=bg,
        )
        frame_tips.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))

        chk_tips = tk.Checkbutton(
            frame_tips,
            text="Mostrar dicas (tooltips)",
            variable=self.tips_var,
            command=self.on_tips_toggle,
            bg=bg,
        )
        chk_tips.pack(anchor="w")

    def on_tips_toggle(self) -> None:
        """Liga/desliga globalmente as dicas (tooltips)."""
        ToolTip.enabled = bool(self.tips_var.get())
        if not ToolTip.enabled:
            for tip in list(ToolTip.instances):
                tip.hidetip()

    def on_center_shift_toggle(self) -> None:
        if self.center_shift_var.get() and not self.center_shift_warned:
            messagebox.showinfo(
                "Centro deslocado",
                "Quando esta opção está ativa, o filtro deixa de estar\n"
                "centrado na frequência 0 e passa a ter um centro (u0, v0)\n"
                "ajustável pelas barras ou clicando na imagem.",
            )
            self.center_shift_warned = True

        self.update_controls_visibility()

        if (
            self.realtime_mask_var.get()
            and self.img_original is not None
            and self.D is not None
        ):
            self.update_mask_preview()

        self.maybe_auto_apply()

    def on_realtime_mask_toggle(self) -> None:
        """Callback da checkbox 'Atualizar máscara em tempo real'."""
        if self.realtime_mask_var.get() and not self.realtime_mask_warned:
            messagebox.showinfo(
                "Aviso",
                "Atualizar a máscara em tempo real pode causar algum atraso (lag).",
            )
            self.realtime_mask_warned = True

        if (
            self.realtime_mask_var.get()
            and self.img_original is not None
            and self.D is not None
        ):
            self.update_mask_preview()

        self.maybe_auto_apply()

    def on_auto_apply_toggle(self) -> None:
        """Callback da checkbox 'Aplicar filtro em tempo real'."""
        if self.auto_apply_var.get() and not self.auto_apply_warned:
            messagebox.showinfo(
                "Aviso",
                "A aplicação automática do filtro ao mudar os parâmetros "
                "pode causar algum atraso (lag), especialmente para imagens grandes.",
            )
            self.auto_apply_warned = True

        if (
            self.auto_apply_var.get()
            and self.img_original is not None
            and self.F_shift is not None
            and self.D is not None
        ):
            self.apply_filter_callback()

    # ------------------------------------------------------------------
    # Utilitários da GUI
    # ------------------------------------------------------------------

    def get_filter_type_code(self) -> str:
        """Devolve 'LPF', 'HPF', 'BPF' ou 'BRF' a partir do texto completo."""
        full = self.filter_type_var.get()
        return full.split()[0]

    def maybe_auto_apply(self) -> None:
        """Aplica o filtro automaticamente se a opção estiver ativa."""
        if self.auto_apply_var.get():
            if (
                self.img_original is not None
                and self.F_shift is not None
                and self.D is not None
            ):
                self.apply_filter_callback()

    def set_title_fixed(self, ax, text: str) -> None:
        ax.set_title(text, y=1.02, fontsize=9)

    def clear_figure(self) -> None:
        axes = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6]
        titles = [
            "Original",
            "Magnitude (orig.)",
            "Fase",
            "Máscara",
            "Magnitude (filtr.)",
            "Filtrada",
        ]
        for ax, t in zip(axes, titles):
            ax.clear()
            self.set_title_fixed(ax, t)
            ax.axis("off")

    # ------------------------------------------------------------------
    # Atualização da máscara (preview)
    # ------------------------------------------------------------------

    def update_mask_preview(self) -> None:
        """Atualiza apenas o plot da máscara (ax4) com base nos parâmetros atuais."""
        if self.D is None:
            return

        filter_type = self.get_filter_type_code()
        profile_type = self.profile_var.get()
        r1 = float(self.scale_r1.get())
        r2 = float(self.scale_r2.get())
        order = int(self.scale_order.get())

        # Para BPF/BRF é obrigatório r1 < r2
        if filter_type in ("BPF", "BRF") and r1 >= r2:
            self.ax4.clear()
            self.set_title_fixed(self.ax4, "Máscara")
            self.ax4.axis("off")
            self.canvas.draw_idle()
            return

        center_offset = None
        if self.center_shift_var.get():
            u0 = float(self.scale_notch_u0.get())
            v0 = float(self.scale_notch_v0.get())
            center_offset = (u0, v0)

        try:
            mask = create_frequency_filter(
                self.D,
                filter_type,
                profile_type,
                r1,
                r2,
                order,
                center_offset=center_offset,
            )
        except Exception:
            return

        self.filter_mask = mask.astype(np.float32)

        self.ax4.clear()
        self.set_title_fixed(self.ax4, "Máscara")
        self.ax4.imshow(self.filter_mask, cmap="gray")
        self.ax4.axis("off")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Callbacks dos sliders r1/r2/ordem/u0/v0
    # ------------------------------------------------------------------

    def on_r1_change(self, value) -> None:
        """Callback do slider r1: garante que em BPF/BRF temos sempre r1 < r2."""
        filter_type = self.get_filter_type_code()
        val_int = int(float(value))
        self.lbl_r1_val.config(text=f"{val_int} px")

        if filter_type in ("BPF", "BRF"):
            r1 = val_int
            r2 = int(self.scale_r2.get())

            if r1 >= r2:
                r2 = r1 + 1
                r2_max = int(self.scale_r2.cget("to"))
                if r2 > r2_max:
                    r2 = r2_max
                r1_min = int(self.scale_r1.cget("from"))
                r1 = max(r2 - 1, r1_min)

                self.scale_r1.set(r1)
                self.lbl_r1_val.config(text=f"{r1} px")
                self.scale_r2.set(r2)
                self.lbl_r2_val.config(text=f"{r2} px")

        if self.realtime_mask_var.get():
            self.update_mask_preview()
        self.maybe_auto_apply()

    def on_r2_change(self, value) -> None:
        """Callback do slider r2: garante que em BPF/BRF temos sempre r1 < r2."""
        filter_type = self.get_filter_type_code()
        val_int = int(float(value))
        self.lbl_r2_val.config(text=f"{val_int} px")

        if filter_type in ("BPF", "BRF"):
            r2 = val_int
            r1 = int(self.scale_r1.get())

            if r2 <= r1:
                r1 = r2 - 1
                r1_min = int(self.scale_r1.cget("from"))
                if r1 < r1_min:
                    r1 = r1_min

                r2_min = int(self.scale_r2.cget("from"))
                if r2 < r2_min:
                    r2 = r2_min

                self.scale_r2.set(r2)
                self.lbl_r2_val.config(text=f"{r2} px")
                self.scale_r1.set(r1)
                self.lbl_r1_val.config(text=f"{r1} px")

        if self.realtime_mask_var.get():
            self.update_mask_preview()
        self.maybe_auto_apply()

    def on_order_change(self, value) -> None:
        """Callback da ordem (Butterworth)."""
        val_int = int(float(value))
        self.lbl_order_val.config(text=f"{val_int}")
        if self.realtime_mask_var.get():
            self.update_mask_preview()
        self.maybe_auto_apply()

    def on_notch_u0_change(self, value) -> None:
        """Callback do centro u0 (quando filtro deslocado está ativo)."""
        val_int = int(float(value))
        self.lbl_u0_val.config(text=f"{val_int} px")
        if self.realtime_mask_var.get():
            self.update_mask_preview()
        self.maybe_auto_apply()

    def on_notch_v0_change(self, value) -> None:
        """Callback do centro v0 (quando filtro deslocado está ativo)."""
        val_int = int(float(value))
        self.lbl_v0_val.config(text=f"{val_int} px")
        if self.realtime_mask_var.get():
            self.update_mask_preview()
        self.maybe_auto_apply()

    # ------------------------------------------------------------------
    # Mostrar / esconder controlos consoante o tipo de filtro
    # ------------------------------------------------------------------

    def update_controls_visibility(self) -> None:
        filter_type = self.get_filter_type_code()
        profile = self.profile_var.get()

        self.r1_frame.pack_forget()
        self.r2_frame.pack_forget()
        self.order_frame.pack_forget()
        self.notch_frame.pack_forget()

        if filter_type == "LPF":
            self.lbl_r2.config(text="Raio exterior r2 [pixels]:")
            self.r2_frame.pack(fill=tk.X, pady=(5, 0))
        elif filter_type == "HPF":
            self.lbl_r1.config(text="Raio interior r1 [pixels]:")
            self.r1_frame.pack(fill=tk.X, pady=(5, 0))
        elif filter_type in ("BPF", "BRF"):
            self.lbl_r1.config(text="Raio interior r1 [pixels]:")
            self.lbl_r2.config(text="Raio exterior r2 [pixels]:")
            self.r1_frame.pack(fill=tk.X, pady=(5, 0))
            self.r2_frame.pack(fill=tk.X, pady=(5, 0))

            r1 = int(self.scale_r1.get())
            r2 = int(self.scale_r2.get())
            if r1 >= r2:
                r1_min = int(self.scale_r1.cget("from"))
                r1 = max(r2 - 1, r1_min)
                self.scale_r1.set(r1)
                self.lbl_r1_val.config(text=f"{r1} px")

        if profile == "Butterworth" and filter_type in ("LPF", "HPF", "BPF", "BRF"):
            self.order_frame.pack(fill=tk.X, pady=(5, 0))

        if self.center_shift_var.get():
            self.notch_frame.pack(fill=tk.X, pady=(5, 0))

    # ------------------------------------------------------------------
    # Clique nas figuras para definir (u0,v0) quando filtro deslocado
    # ------------------------------------------------------------------

    def on_figure_click(self, event) -> None:
        if self.img_original is None or self.D is None:
            return
        if not self.center_shift_var.get():
            return
        if event.inaxes not in (self.ax1, self.ax2, self.ax3, self.ax5, self.ax6):
            return
        if event.xdata is None or event.ydata is None:
            return

        rows, cols = self.D.shape
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        if col < 0 or col >= cols or row < 0 or row >= rows:
            return

        crow, ccol = rows // 2, cols // 2
        u0 = col - ccol
        v0 = row - crow

        u_min = float(self.scale_notch_u0.cget("from"))
        u_max = float(self.scale_notch_u0.cget("to"))
        v_min = float(self.scale_notch_v0.cget("from"))
        v_max = float(self.scale_notch_v0.cget("to"))

        u0 = max(min(u0, u_max), u_min)
        v0 = max(min(v0, v_max), v_min)

        self.scale_notch_u0.set(int(u0))
        self.scale_notch_v0.set(int(v0))

        if self.realtime_mask_var.get():
            self.update_mask_preview()
        self.maybe_auto_apply()

    # ------------------------------------------------------------------
    # Callbacks principais da GUI
    # ------------------------------------------------------------------

    def load_image_callback(self) -> None:
        filetypes = [
            ("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("Todos os ficheiros", "*.*"),
        ]
        filepath = filedialog.askopenfilename(
            title="Selecionar imagem",
            filetypes=filetypes,
        )
        if not filepath:
            return

        try:
            img_arr, mode = load_image_any(filepath)
        except Exception as e:
            messagebox.showerror("Erro ao carregar imagem", str(e))
            return

        if mode == "rgb":
            use_intensity = messagebox.askyesno(
                "Imagem RGB detetada",
                "A imagem carregada é RGB.\n\n"
                "Se escolheres 'Sim', a imagem será convertida para intensidade.\n"
                "Se escolheres 'Não', o filtro será aplicado de igual forma "
                "às 3 componentes RGB.",
            )
            if use_intensity:
                img_gray = (
                    0.299 * img_arr[..., 0]
                    + 0.587 * img_arr[..., 1]
                    + 0.114 * img_arr[..., 2]
                ).astype(np.float32)
                self.img_original = img_gray
            else:
                self.img_original = img_arr.astype(np.float32)
        else:
            self.img_original = img_arr.astype(np.float32)

        self.img_filtered = None
        self.filter_mask = None
        self.img_path = filepath

        rows, cols = self.img_original.shape[:2]
        self.D = precompute_distance_map((rows, cols))

        F = np.fft.fft2(self.img_original, axes=(0, 1))
        self.F_shift = np.fft.fftshift(F, axes=(0, 1))

        self.magnitude_original = compute_magnitude(self.F_shift)
        self.phase_original = compute_phase(self.F_shift)
        self.magnitude_filtered = None

        r_max = max(1, min(rows, cols) // 2)
        self.scale_r1.config(from_=0, to=r_max)
        self.scale_r2.config(from_=1, to=r_max)

        self.scale_r1.set(r_max // 10)
        self.lbl_r1_val.config(text=f"{self.scale_r1.get()} px")
        self.scale_r2.set(r_max // 4)
        self.lbl_r2_val.config(text=f"{self.scale_r2.get()} px")

        u_max = cols // 2
        v_max = rows // 2
        self.scale_notch_u0.config(from_=-u_max, to=u_max)
        self.scale_notch_v0.config(from_=-v_max, to=v_max)

        self.clear_figure()

        if self.img_original.ndim == 2:
            self.ax1.imshow(self.img_original, cmap="gray", vmin=0, vmax=1)
        else:
            self.ax1.imshow(self.img_original, vmin=0, vmax=1)
        self.ax1.axis("off")

        if self.magnitude_original is not None:
            self.ax2.imshow(self.magnitude_original, cmap="gray")
            self.ax2.axis("off")

        if self.phase_original is not None:
            self.ax3.imshow(self.phase_original, cmap="gray")
            self.ax3.axis("off")

        self.canvas.draw()

        self.update_metrics_label()

        if self.realtime_mask_var.get():
            self.update_mask_preview()

        self.maybe_auto_apply()

    def apply_filter_callback(self) -> None:
        if self.img_original is None or self.F_shift is None or self.D is None:
            messagebox.showwarning(
                "Aviso", "Carrega primeiro uma imagem antes de aplicar o filtro."
            )
            return

        filter_type = self.get_filter_type_code()
        profile_type = self.profile_var.get()
        r1 = float(self.scale_r1.get())
        r2 = float(self.scale_r2.get())
        order = int(self.scale_order.get())

        if filter_type in ("BPF", "BRF") and r1 >= r2:
            messagebox.showwarning(
                "Parâmetros inválidos",
                "Para BPF/BRF é necessário r1 < r2.",
            )
            return

        center_offset = None
        if self.center_shift_var.get():
            u0 = float(self.scale_notch_u0.get())
            v0 = float(self.scale_notch_v0.get())
            center_offset = (u0, v0)

        try:
            mask = create_frequency_filter(
                self.D,
                filter_type,
                profile_type,
                r1,
                r2,
                order,
                center_offset=center_offset,
            )
        except Exception as e:
            messagebox.showerror("Erro ao criar filtro", str(e))
            return

        if self.F_shift.ndim == 3:
            F_filtered_shift = self.F_shift * mask[:, :, np.newaxis]
        else:
            F_filtered_shift = self.F_shift * mask

        F_filtered = np.fft.ifftshift(F_filtered_shift, axes=(0, 1))
        img_filtered_complex = np.fft.ifft2(F_filtered, axes=(0, 1))
        img_filtered = np.real(img_filtered_complex).astype(np.float32)

        img_filtered -= img_filtered.min()
        max_val = img_filtered.max()
        if max_val > 0:
            img_filtered /= max_val

        self.img_filtered = img_filtered
        self.filter_mask = mask.astype(np.float32)
        self.magnitude_filtered = compute_magnitude(F_filtered_shift)

        self.clear_figure()

        if self.img_original.ndim == 2:
            self.ax1.imshow(self.img_original, cmap="gray", vmin=0, vmax=1)
        else:
            self.ax1.imshow(self.img_original, vmin=0, vmax=1)
        self.ax1.axis("off")

        if self.magnitude_original is not None:
            self.ax2.imshow(self.magnitude_original, cmap="gray")
            self.ax2.axis("off")

        if self.phase_original is not None:
            self.ax3.imshow(self.phase_original, cmap="gray")
            self.ax3.axis("off")

        if self.filter_mask is not None:
            self.ax4.imshow(self.filter_mask, cmap="gray")
            self.ax4.axis("off")

        if self.magnitude_filtered is not None:
            self.ax5.imshow(self.magnitude_filtered, cmap="gray")
            self.ax5.axis("off")

        if self.img_filtered.ndim == 2:
            self.ax6.imshow(self.img_filtered, cmap="gray", vmin=0, vmax=1)
        else:
            self.ax6.imshow(self.img_filtered, vmin=0, vmax=1)
        self.ax6.axis("off")

        self.canvas.draw()
        self.update_metrics_label()

    def update_metrics_label(self) -> None:
        if self.img_original is None or self.img_filtered is None:
            self.metrics_label.config(text="MSE: --\nPSNR: -- dB")
            return

        try:
            m = mse(self.img_original, self.img_filtered)
            p = psnr(self.img_original, self.img_filtered)
        except Exception:
            self.metrics_label.config(text="MSE: erro\nPSNR: erro")
            return

        psnr_str = "Inf" if np.isinf(p) else f"{p:.2f}"
        self.metrics_label.config(text=f"MSE: {m:.6f}\nPSNR: {psnr_str} dB")

    def save_image_callback(self) -> None:
        if self.img_filtered is None:
            messagebox.showwarning("Aviso", "Nenhuma imagem filtrada para guardar.")
            return

        suggested_name = "imagem_filtrada"
        if self.img_path:
            base, _ = os.path.splitext(os.path.basename(self.img_path))
            suggested_name = f"{base}_filtrado"

        filepath = filedialog.asksaveasfilename(
            initialfile=suggested_name,
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg;*.jpeg"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tif;*.tiff"),
                ("Todos os ficheiros", "*.*"),
            ],
            title="Guardar imagem filtrada",
        )
        if not filepath:
            return

        extras_flags = [
            self.save_orig_var.get(),
            self.save_mag_orig_var.get(),
            self.save_phase_var.get(),
            self.save_mask_var.get(),
            self.save_mag_filt_var.get(),
            self.save_specs_var.get(),
        ]
        extras_selected = any(extras_flags)

        dirpath, filename = os.path.split(filepath)
        base_name, ext = os.path.splitext(filename)

        if extras_selected:
            out_dir = os.path.join(dirpath, base_name)
            os.makedirs(out_dir, exist_ok=True)
            filtered_path = os.path.join(out_dir, filename)
        else:
            out_dir = dirpath
            filtered_path = filepath

        img_out = (self.img_filtered * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_out).save(filtered_path)
        created_files = [filtered_path]

        if self.save_orig_var.get() and self.img_original is not None:
            path = os.path.join(out_dir, f"{base_name}_original{ext}")
            save_array_as_png(self.img_original, path)
            created_files.append(path)

        if self.save_mag_orig_var.get() and self.magnitude_original is not None:
            path = os.path.join(out_dir, f"{base_name}_mag_orig{ext}")
            save_array_as_png(self.magnitude_original, path)
            created_files.append(path)

        if self.save_phase_var.get() and self.phase_original is not None:
            path = os.path.join(out_dir, f"{base_name}_fase{ext}")
            save_array_as_png(self.phase_original, path)
            created_files.append(path)

        if self.save_mask_var.get() and self.filter_mask is not None:
            path = os.path.join(out_dir, f"{base_name}_mascara{ext}")
            save_array_as_png(self.filter_mask, path)
            created_files.append(path)

        if self.save_mag_filt_var.get() and self.magnitude_filtered is not None:
            path = os.path.join(out_dir, f"{base_name}_mag_filtr{ext}")
            save_array_as_png(self.magnitude_filtered, path)
            created_files.append(path)

        if self.save_specs_var.get():
            m = (
                mse(self.img_original, self.img_filtered)
                if self.img_original is not None
                else float("nan")
            )
            p = (
                psnr(self.img_original, self.img_filtered)
                if self.img_original is not None
                else float("nan")
            )

            filter_type_full = self.filter_type_var.get()
            filter_type = self.get_filter_type_code()
            profile_type = self.profile_var.get()
            r1 = float(self.scale_r1.get())
            r2 = float(self.scale_r2.get())
            order = int(self.scale_order.get())
            u0 = float(self.scale_notch_u0.get())
            v0 = float(self.scale_notch_v0.get())

            specs_path = os.path.join(out_dir, f"{base_name}_spcs.txt")
            with open(specs_path, "w", encoding="utf-8") as f:
                f.write("=== Filtro no domínio da frequência ===\n\n")
                f.write(f"Imagem filtrada: {filtered_path}\n\n")
                f.write(f"Tipo de filtro: {filter_type_full} ({filter_type})\n")
                f.write(f"Perfil: {profile_type}\n")
                if filter_type in ("HPF", "BPF", "BRF"):
                    f.write(f"Raio interior (r1): {r1:.2f}\n")
                if filter_type in ("LPF", "BPF", "BRF"):
                    f.write(f"Raio exterior (r2): {r2:.2f}\n")
                if self.center_shift_var.get():
                    f.write(f"Centro (u0, v0): ({u0:.2f}, {v0:.2f})\n")
                if profile_type == "Butterworth" and filter_type in (
                    "LPF",
                    "HPF",
                    "BPF",
                    "BRF",
                ):
                    f.write(f"Ordem: {order}\n")
                f.write("\n")
                f.write(f"MSE: {m:.6f}\n")
                f.write(f"PSNR: {'Inf' if np.isinf(p) else f'{p:.2f} dB'}\n")

            created_files.append(specs_path)

        msg = "Ficheiros criados:\n" + "\n".join(created_files)
        messagebox.showinfo("Guardar", msg)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> None:
    root = tk.Tk()
    
    root.state("zoomed")
    
    FrequencyFilterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
