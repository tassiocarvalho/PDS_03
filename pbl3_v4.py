"""
Aplicativo de Projeto de Filtros por Janelamento (Versão Final Corrigida)
Desenvolvido para o Problema 03 de PDS

Este aplicativo permite projetar filtros digitais usando a técnica de janelamento,
com interface gráfica amigável para entrada de especificações, visualização de
resultados e interação com o usuário.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, messagebox
from scipy import signal
import matplotlib
matplotlib.use("TkAgg")

class FilterDesignApp:
    """
    Aplicativo para projeto de filtros digitais usando técnica de janelamento.
    
    Características:
    - Interface gráfica amigável
    - Múltiplas funções de janelamento
    - Diferentes tipos de filtros (passa-baixa, passa-alta, passa-faixa, rejeita-faixa)
    - Projeto automático com janela de Kaiser
    - Visualização gráfica dos resultados
    - Métricas detalhadas do filtro
    """
    
    def __init__(self, root):
        """
        Inicializa a aplicação e configura a interface gráfica.
        
        Args:
            root: Janela principal do Tkinter
        """
        self.root = root
        self.root.title("Projeto de Filtros por Janelamento - Versão Final")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configuração de estilo
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Variáveis de controle
        self.filter_type = tk.StringVar(value="passa-baixa")
        self.window_type = tk.StringVar(value="hamming")
        self.cutoff_freq_str = tk.StringVar(value="0.20")
        self.cutoff_freq2_str = tk.StringVar(value="0.40")
        self.filter_order = tk.IntVar(value=51)
        self.freq_step = 0.01
        
        # Variáveis para projeto Kaiser
        self.delta_var = tk.StringVar(value="0.01")
        self.wp_var = tk.StringVar(value="0.4")
        self.ws_var = tk.StringVar(value="0.6")
        
        # Variáveis para frequência de amostragem e unidade
        self.fs_var = tk.StringVar(value="1000")  # Frequência de amostragem em Hz
        self.freq_unit = tk.StringVar(value="normalizada")  # "normalizada" ou "hz"
        
        # Criar frames principais
        self.create_frames()
        
        # Configurar área de visualização
        self.setup_visualization()
        
        # Configurar widgets de entrada
        self.setup_input_widgets()
        
        # Inicializar com valores padrão
        self.update_filter()
    
    def create_frames(self):
        """Cria os frames principais da interface"""
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Especificações do Filtro", padding=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        self.viz_frame = ttk.LabelFrame(self.main_frame, text="Visualização", padding=10)
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def setup_input_widgets(self):
        """Configura os widgets de entrada para as especificações do filtro"""
        row_idx = 0
        
        # Tipo de filtro
        ttk.Label(self.control_frame, text="Tipo de Filtro:").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        filter_types = ["passa-baixa", "passa-alta", "passa-faixa", "rejeita-faixa"]
        ttk.OptionMenu(self.control_frame, self.filter_type, filter_types[0], *filter_types,
                      command=self.on_filter_type_change).grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=5)
        row_idx += 1
        
        # Tipo de janela
        ttk.Label(self.control_frame, text="Função de Janelamento:").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        window_types = ["retangular", "hamming", "hanning", "blackman", "bartlett", "kaiser"]
        ttk.OptionMenu(self.control_frame, self.window_type, window_types[1], *window_types,
                      command=self.on_window_type_change).grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW, pady=5)
        row_idx += 1
        
        # Parâmetro beta para Kaiser (inicialmente oculto)
        self.beta_label = ttk.Label(self.control_frame, text="Parâmetro β (Kaiser):")
        self.beta_var = tk.StringVar(value="8.0")
        self.beta_frame = ttk.Frame(self.control_frame)
        ttk.Button(self.beta_frame, text="-", width=2, command=lambda: self.adjust_beta(-0.5)).pack(side=tk.LEFT, padx=(0, 2))
        self.beta_entry = ttk.Entry(self.beta_frame, textvariable=self.beta_var, width=5, justify=tk.CENTER)
        self.beta_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(self.beta_frame, text="+", width=2, command=lambda: self.adjust_beta(0.5)).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(self.beta_frame, text="(0.0-15.0)").pack(side=tk.LEFT, padx=5)
        self.beta_entry.bind("<Return>", lambda _: self.validate_beta_and_update())
        self.beta_entry.bind("<FocusOut>", lambda _: self.validate_beta_and_update())
        # Inicialmente oculto
        row_idx += 1
        
        # Configuração de unidade de frequência
        freq_unit_frame = ttk.LabelFrame(self.control_frame, text="Unidade de Frequência", padding=5)
        freq_unit_frame.grid(row=row_idx, column=0, columnspan=4, sticky=tk.EW, pady=5)
        
        # Radio buttons para escolher unidade
        unit_frame = ttk.Frame(freq_unit_frame)
        unit_frame.pack(fill=tk.X, pady=2)
        
        ttk.Radiobutton(unit_frame, text="Normalizada (×π)", variable=self.freq_unit, 
                       value="normalizada", command=self.on_freq_unit_change).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(unit_frame, text="Hz", variable=self.freq_unit, 
                       value="hz", command=self.on_freq_unit_change).pack(side=tk.LEFT, padx=15)
        
        # Frame para frequência de amostragem (inicialmente oculto)
        self.fs_frame = ttk.Frame(freq_unit_frame)
        ttk.Label(self.fs_frame, text="Fs (Hz):").pack(side=tk.LEFT)
        fs_entry = ttk.Entry(self.fs_frame, textvariable=self.fs_var, width=8)
        fs_entry.pack(side=tk.LEFT, padx=5)
        fs_entry.bind("<Return>", lambda _: self.on_fs_change())
        fs_entry.bind("<FocusOut>", lambda _: self.on_fs_change())
        
        row_idx += 1
        
        # Frequência de corte 1 com botões
        self.freq1_label = ttk.Label(self.control_frame, text="Freq. Corte 1 (ωc1):")
        self.freq1_label.grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        freq1_frame = ttk.Frame(self.control_frame)
        freq1_frame.grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW)
        ttk.Button(freq1_frame, text="-", width=2, command=lambda: self.adjust_freq(self.cutoff_freq_str, -self.get_freq_step())).pack(side=tk.LEFT, padx=(0, 2))
        self.cutoff_entry1 = ttk.Entry(freq1_frame, textvariable=self.cutoff_freq_str, width=8, justify=tk.CENTER)
        self.cutoff_entry1.pack(side=tk.LEFT, padx=2)
        ttk.Button(freq1_frame, text="+", width=2, command=lambda: self.adjust_freq(self.cutoff_freq_str, self.get_freq_step())).pack(side=tk.LEFT, padx=(2, 0))
        self.freq1_unit_label = ttk.Label(freq1_frame, text="π (0.01-0.99)")
        self.freq1_unit_label.pack(side=tk.LEFT, padx=5)
        self.cutoff_entry1.bind("<Return>", lambda _: self.validate_and_update())
        self.cutoff_entry1.bind("<FocusOut>", lambda _: self.validate_and_update())
        row_idx += 1
        
        # Frequência de corte 2 com botões
        self.cutoff2_label_widget = ttk.Label(self.control_frame, text="Freq. Corte 2 (ωc2):")
        self.cutoff2_label_widget.grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        self.freq2_frame = ttk.Frame(self.control_frame)
        self.freq2_frame.grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW)
        ttk.Button(self.freq2_frame, text="-", width=2, command=lambda: self.adjust_freq(self.cutoff_freq2_str, -self.get_freq_step())).pack(side=tk.LEFT, padx=(0, 2))
        self.cutoff_entry2 = ttk.Entry(self.freq2_frame, textvariable=self.cutoff_freq2_str, width=8, justify=tk.CENTER)
        self.cutoff_entry2.pack(side=tk.LEFT, padx=2)
        ttk.Button(self.freq2_frame, text="+", width=2, command=lambda: self.adjust_freq(self.cutoff_freq2_str, self.get_freq_step())).pack(side=tk.LEFT, padx=(2, 0))
        self.freq2_unit_label = ttk.Label(self.freq2_frame, text="π (0.01-0.99)")
        self.freq2_unit_label.pack(side=tk.LEFT, padx=5)
        self.cutoff_entry2.bind("<Return>", lambda _: self.validate_and_update())
        self.cutoff_entry2.bind("<FocusOut>", lambda _: self.validate_and_update())
        row_idx += 1
        
        # Ordem do filtro (número de coeficientes)
        ttk.Label(self.control_frame, text="Ordem do Filtro:").grid(row=row_idx, column=0, sticky=tk.W, pady=5)
        order_frame = ttk.Frame(self.control_frame)
        order_frame.grid(row=row_idx, column=1, columnspan=3, sticky=tk.EW)
        ttk.Button(order_frame, text="-", width=2, command=lambda: self.adjust_order(-2)).pack(side=tk.LEFT, padx=(0, 2))
        self.order_entry = ttk.Entry(order_frame, textvariable=tk.StringVar(value=str(self.filter_order.get())), width=5, justify=tk.CENTER)
        self.order_entry.pack(side=tk.LEFT, padx=2)
        ttk.Button(order_frame, text="+", width=2, command=lambda: self.adjust_order(2)).pack(side=tk.LEFT, padx=(2, 0))
        ttk.Label(order_frame, text="(ímpar, 11-201)").pack(side=tk.LEFT, padx=5)
        self.order_entry.bind("<Return>", lambda _: self.validate_order_and_update())
        self.order_entry.bind("<FocusOut>", lambda _: self.validate_order_and_update())
        row_idx += 1
        
        # Visualização de fase
        self.show_compensated_phase = tk.BooleanVar(value=True)
        phase_frame = ttk.Frame(self.control_frame)
        phase_frame.grid(row=row_idx, column=0, columnspan=4, sticky=tk.EW, pady=5)
        ttk.Label(phase_frame, text="Visualização de Fase:").pack(side=tk.LEFT)
        ttk.Radiobutton(phase_frame, text="Compensada", variable=self.show_compensated_phase, 
                       value=True, command=self.update_filter).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(phase_frame, text="Original", variable=self.show_compensated_phase, 
                       value=False, command=self.update_filter).pack(side=tk.LEFT, padx=5)
        row_idx += 1
        
        # Separador
        ttk.Separator(self.control_frame, orient='horizontal').grid(row=row_idx, column=0, columnspan=4, sticky="ew", pady=10)
        row_idx += 1
        
        # Projeto automático com Kaiser  
        kaiser_frame = ttk.LabelFrame(self.control_frame, text="Projeto Automático (Kaiser)", padding=5)
        kaiser_frame.grid(row=row_idx, column=0, columnspan=4, sticky=tk.EW, pady=5)
        
        # Especificações para Kaiser - linha 1
        spec_frame1 = ttk.Frame(kaiser_frame)
        spec_frame1.pack(fill=tk.X, pady=2)
        
        ttk.Label(spec_frame1, text="δ (erro):").pack(side=tk.LEFT)
        delta_entry = ttk.Entry(spec_frame1, textvariable=self.delta_var, width=8)
        delta_entry.pack(side=tk.LEFT, padx=(5, 15))
        
        ttk.Label(spec_frame1, text="ωp:").pack(side=tk.LEFT)
        wp_entry = ttk.Entry(spec_frame1, textvariable=self.wp_var, width=6)
        wp_entry.pack(side=tk.LEFT, padx=5)
        
        # Especificações para Kaiser - linha 2
        spec_frame2 = ttk.Frame(kaiser_frame)
        spec_frame2.pack(fill=tk.X, pady=2)
        
        ttk.Label(spec_frame2, text="ωs:").pack(side=tk.LEFT)
        ws_entry = ttk.Entry(spec_frame2, textvariable=self.ws_var, width=6)
        ws_entry.pack(side=tk.LEFT, padx=(32, 15))
        
        # Botão para calcular
        ttk.Button(spec_frame2, text="Calcular Kaiser", 
                  command=self.design_kaiser_filter).pack(side=tk.LEFT, padx=10)
        row_idx += 1
        
        # Informações adicionais
        info_frame = ttk.LabelFrame(self.control_frame, text="Informações e Métricas", padding=10)
        info_frame.grid(row=row_idx, column=0, columnspan=4, sticky=tk.NSEW, pady=10)
        self.control_frame.grid_rowconfigure(row_idx, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=1)
        
        self.info_text = tk.Text(info_frame, height=15, width=45, wrap=tk.WORD, font=('Consolas', 9))
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=scrollbar.set)
        
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.info_text.insert(tk.END, "Informações sobre o filtro aparecerão aqui.")
        self.info_text.config(state=tk.DISABLED)
        
        # Inicialmente esconder a segunda frequência de corte e controles Kaiser
        self.toggle_cutoff2_visibility()
        self.toggle_kaiser_controls()
        self.update_freq_labels()  # Configurar labels iniciais

    def get_freq_step(self):
        """Retorna o passo de frequência apropriado baseado na unidade atual"""
        if self.freq_unit.get() == "hz":
            fs = float(self.fs_var.get())
            return fs * 0.01  # 1% da frequência de amostragem
        else:
            return 0.01  # Passo normalizado padrão

    def on_freq_unit_change(self):
        """Callback quando a unidade de frequência é alterada"""
        if self.freq_unit.get() == "hz":
            # Mostrar campo Fs e converter frequências normalizadas para Hz
            self.fs_frame.pack(fill=tk.X, pady=2)
            self.convert_freqs_to_hz()
        else:
            # Esconder campo Fs e converter frequências Hz para normalizadas
            self.fs_frame.pack_forget()
            self.convert_freqs_to_normalized()
        
        self.update_freq_labels()
        self.update_kaiser_labels()

    def on_fs_change(self):
        """Callback quando a frequência de amostragem é alterada"""
        try:
            fs = float(self.fs_var.get())
            if fs <= 0:
                raise ValueError("Fs deve ser positiva")
            # Se estamos em modo Hz, reconverter as frequências
            if self.freq_unit.get() == "hz":
                self.update_freq_labels()
        except ValueError:
            messagebox.showerror("Erro", "Frequência de amostragem inválida")
            self.fs_var.set("1000")

    def convert_freqs_to_hz(self):
        """Converte frequências normalizadas para Hz"""
        try:
            fs = float(self.fs_var.get())
            
            # Converter frequência de corte 1
            freq1_norm = float(self.cutoff_freq_str.get())
            freq1_hz = freq1_norm * fs / 2  # Nyquist = fs/2
            self.cutoff_freq_str.set(f"{freq1_hz:.1f}")
            
            # Converter frequência de corte 2 se necessário
            if self.filter_type.get() in ["passa-faixa", "rejeita-faixa"]:
                freq2_norm = float(self.cutoff_freq2_str.get())
                freq2_hz = freq2_norm * fs / 2
                self.cutoff_freq2_str.set(f"{freq2_hz:.1f}")
                
        except ValueError:
            pass  # Manter valores atuais se conversão falhar

    def convert_freqs_to_normalized(self):
        """Converte frequências Hz para normalizadas"""
        try:
            fs = float(self.fs_var.get())
            
            # Converter frequência de corte 1
            freq1_hz = float(self.cutoff_freq_str.get())
            freq1_norm = freq1_hz * 2 / fs  # Normalizar por Nyquist
            freq1_norm = max(0.01, min(0.99, freq1_norm))  # Limitar
            self.cutoff_freq_str.set(f"{freq1_norm:.2f}")
            
            # Converter frequência de corte 2 se necessário
            if self.filter_type.get() in ["passa-faixa", "rejeita-faixa"]:
                freq2_hz = float(self.cutoff_freq2_str.get())
                freq2_norm = freq2_hz * 2 / fs
                freq2_norm = max(0.01, min(0.99, freq2_norm))
                self.cutoff_freq2_str.set(f"{freq2_norm:.2f}")
                
        except ValueError:
            pass  # Manter valores atuais se conversão falhar

    def update_freq_labels(self):
        """Atualiza os labels das frequências baseado na unidade atual"""
        if self.freq_unit.get() == "hz":
            fs = float(self.fs_var.get())
            nyquist = fs / 2
            self.freq1_label.config(text="Freq. Corte 1 (fc1):")
            self.cutoff2_label_widget.config(text="Freq. Corte 2 (fc2):")
            self.freq1_unit_label.config(text=f"Hz (1-{nyquist:.0f})")
            self.freq2_unit_label.config(text=f"Hz (1-{nyquist:.0f})")
        else:
            self.freq1_label.config(text="Freq. Corte 1 (ωc1):")
            self.cutoff2_label_widget.config(text="Freq. Corte 2 (ωc2):")
            self.freq1_unit_label.config(text="π (0.01-0.99)")
            self.freq2_unit_label.config(text="π (0.01-0.99)")

    def update_kaiser_labels(self):
        """Atualiza os labels do projeto Kaiser baseado na unidade atual"""
        # Os labels ωp e ωs no projeto Kaiser sempre usam normalização
        # mas vamos manter consistência visual
        pass  # Kaiser sempre trabalha internamente com frequências normalizadas

    def adjust_beta(self, delta):
        """Ajusta o parâmetro beta da janela Kaiser"""
        try:
            current_val = float(self.beta_var.get())
            new_val = round(current_val + delta, 1)
            new_val = max(0.0, min(15.0, new_val))
            self.beta_var.set(f"{new_val:.1f}")
            if self.window_type.get() == "kaiser":
                self.update_filter()
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Valor de β inválido.")
            self.beta_var.set("8.0")

    def validate_beta_and_update(self):
        """Valida o parâmetro beta e atualiza o filtro"""
        try:
            val = float(self.beta_var.get())
            if 0.0 <= val <= 15.0:
                self.beta_var.set(f"{val:.1f}")
                if self.window_type.get() == "kaiser":
                    self.update_filter()
            else:
                messagebox.showerror("Erro de Entrada", "β deve estar entre 0.0 e 15.0.")
                self.beta_var.set("8.0")
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Valor de β inválido.")
            self.beta_var.set("8.0")

    def toggle_kaiser_controls(self):
        """Mostra ou esconde os controles específicos da janela Kaiser"""
        if self.window_type.get() == "kaiser":
            # Encontrar a posição correta para inserir os controles Kaiser
            kaiser_row = 3  # Após o tipo de janela
            self.beta_label.grid(row=kaiser_row, column=0, sticky=tk.W, pady=5)
            self.beta_frame.grid(row=kaiser_row, column=1, columnspan=3, sticky=tk.EW)
        else:
            self.beta_label.grid_remove()
            self.beta_frame.grid_remove()

    def on_window_type_change(self, _):
        """Callback quando o tipo de janela é alterado"""
        self.toggle_kaiser_controls()
        self.update_filter()

    def design_kaiser_filter(self):
        """
        Projeta filtro usando as fórmulas de Kaiser (Equações 7.75 e 7.76)
        Implementação exata conforme Seção 7.5.3 do livro
        """
        try:
            # Obter especificações
            delta = float(self.delta_var.get())
            wp = float(self.wp_var.get()) * np.pi  # converter para radianos
            ws = float(self.ws_var.get()) * np.pi  # converter para radianos
            
            if delta <= 0 or delta >= 1:
                raise ValueError("δ deve estar entre 0 e 1")
            if wp >= ws:
                raise ValueError("ωp deve ser menor que ωs")
            if wp <= 0 or ws >= np.pi:
                raise ValueError("Frequências devem estar entre 0 e π")
            
            # Calcular parâmetros conforme Seção 7.6.1
            A = -20 * np.log10(delta)  # Equação 7.74
            delta_omega = ws - wp  # Largura de transição em radianos
            
            # Equação 7.75 - Calcular β
            if A > 50:
                beta = 0.1102 * (A - 8.7)
            elif 21 <= A <= 50:
                beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
            else:
                beta = 0.0
            
            # Equação 7.76 - Calcular M (usar Δω em radianos)
            M = (A - 8) / (2.285 * delta_omega)
            M = int(np.ceil(M))  # Arredondar para cima
            
            # Garantir que M seja ímpar para fase linear
            if M % 2 == 0:
                M += 1
            
            # Limitar M ao intervalo permitido
            M = max(11, min(201, M))
            
            # Calcular frequência de corte (centro da transição)
            wc = (wp + ws) / 2
            
            # Atualizar interface
            self.filter_order.set(M)
            self.order_entry.delete(0, tk.END)
            self.order_entry.insert(0, str(M))
            
            self.cutoff_freq_str.set(f"{wc/np.pi:.3f}")
            self.window_type.set("kaiser")
            self.beta_var.set(f"{beta:.3f}")  # Atualizar também o β calculado
            
            # Mostrar/ocultar controles Kaiser
            self.toggle_kaiser_controls()
            
            # Mostrar resultados
            result_msg = (
                f"Projeto Kaiser Concluído:\n\n"
                f"Especificações:\n"
                f"• δ = {delta:.4f}\n"
                f"• ωp = {wp/np.pi:.3f}π\n"
                f"• ωs = {ws/np.pi:.3f}π\n\n"
                f"Parâmetros Calculados:\n"
                f"• A = {A:.1f} dB\n"
                f"• Δω = {delta_omega/np.pi:.3f}π\n"
                f"• β = {beta:.3f}\n"
                f"• M = {M}\n"
                f"• ωc = {wc/np.pi:.3f}π\n\n"
                f"Filtro atualizado automaticamente!"
            )
            messagebox.showinfo("Projeto Kaiser", result_msg)
            
            # Atualizar filtro
            self.update_filter()
            
        except ValueError as e:
            messagebox.showerror("Erro", f"Erro nos parâmetros: {e}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro no projeto: {e}")

    def setup_visualization(self):
        """Configura a área de visualização com múltiplos gráficos"""
        self.notebook = ttk.Notebook(self.viz_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.window_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.window_tab, text="Função de Janelamento")
        
        self.coef_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.coef_tab, text="Coeficientes do Filtro")
        
        self.freq_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.freq_tab, text="Resposta em Frequência")
        
        self.setup_window_plot()
        self.setup_coef_plot()
        self.setup_freq_plot()
    
    def setup_window_plot(self):
        """Configura o gráfico para visualização da função de janelamento"""
        self.window_fig = Figure(figsize=(8, 6), dpi=100)
        self.window_canvas = FigureCanvasTkAgg(self.window_fig, master=self.window_tab)
        self.window_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.window_canvas, self.window_tab)
        toolbar.update()
    
    def setup_coef_plot(self):
        """Configura o gráfico para visualização dos coeficientes do filtro"""
        self.coef_fig = Figure(figsize=(8, 6), dpi=100)
        self.coef_canvas = FigureCanvasTkAgg(self.coef_fig, master=self.coef_tab)
        self.coef_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.coef_canvas, self.coef_tab)
        toolbar.update()
    
    def setup_freq_plot(self):
        """Configura o gráfico para visualização da resposta em frequência"""
        self.freq_fig = Figure(figsize=(8, 6), dpi=100)
        self.freq_canvas = FigureCanvasTkAgg(self.freq_fig, master=self.freq_tab)
        self.freq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.freq_canvas, self.freq_tab)
        toolbar.update()

    def adjust_freq(self, freq_var, delta):
        """Ajusta a frequência de corte usando os botões +/-."""
        try:
            current_val = float(freq_var.get())
            new_val = current_val + delta
            
            if self.freq_unit.get() == "hz":
                # Limitar entre 1 Hz e Nyquist
                fs = float(self.fs_var.get())
                nyquist = fs / 2
                new_val = max(1, min(nyquist - 1, new_val))
                freq_var.set(f"{new_val:.1f}")
            else:
                # Limitar frequência normalizada
                new_val = max(0.01, min(0.99, round(new_val, 2)))
                freq_var.set(f"{new_val:.2f}")
                
            self.validate_and_update()
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Valor de frequência inválido.")
            if self.freq_unit.get() == "hz":
                freq_var.set("200")
            else:
                freq_var.set("0.20") 

    def adjust_order(self, delta):
        """Ajusta a ordem do filtro usando os botões +/-."""
        try:
            current_val = self.filter_order.get()
            new_val = current_val + delta
            if new_val % 2 == 0:
                new_val += np.sign(delta) if delta != 0 else 1
            new_val = max(11, min(201, new_val))
            self.filter_order.set(new_val)
            self.order_entry.delete(0, tk.END)
            self.order_entry.insert(0, str(new_val))
            self.update_filter()
        except ValueError:
             messagebox.showerror("Erro de Entrada", "Valor de ordem inválido.")
             self.filter_order.set(51)
             self.order_entry.delete(0, tk.END)
             self.order_entry.insert(0, "51")

    def validate_freq(self, freq_str):
        """Valida a string de frequência de corte baseada na unidade atual."""
        try:
            val = float(freq_str)
            
            if self.freq_unit.get() == "hz":
                fs = float(self.fs_var.get())
                nyquist = fs / 2
                if 1 <= val < nyquist:
                    return val
                else:
                    messagebox.showerror("Erro de Entrada", f"Frequência deve estar entre 1 Hz e {nyquist:.0f} Hz.")
                    return None
            else:
                if 0.01 <= val <= 0.99:
                    return val
                else:
                    messagebox.showerror("Erro de Entrada", "Frequência normalizada deve estar entre 0.01 e 0.99.")
                    return None
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Valor de frequência inválido.")
            return None

    def freq_to_normalized(self, freq_val):
        """Converte frequência para forma normalizada (sempre entre 0 e 1)"""
        if self.freq_unit.get() == "hz":
            fs = float(self.fs_var.get())
            return freq_val * 2 / fs  # Normalizar por Nyquist
        else:
            return freq_val  # Já normalizada

    def validate_order(self, order_str):
        """Valida a string da ordem do filtro."""
        try:
            val = int(order_str)
            if 11 <= val <= 201:
                if val % 2 == 0:
                    messagebox.showwarning("Ajuste de Ordem", "Ordem do filtro deve ser ímpar para fase linear. Ajustando para o próximo ímpar.")
                    val += 1
                    val = min(201, val)
                return val
            else:
                messagebox.showerror("Erro de Entrada", "Ordem do filtro deve estar entre 11 e 201.")
                return None
        except ValueError:
            messagebox.showerror("Erro de Entrada", "Valor de ordem inválido.")
            return None

    def validate_and_update(self):
        """Valida todas as entradas e atualiza o filtro se válido."""
        wc1 = self.validate_freq(self.cutoff_freq_str.get())
        if wc1 is None:
            if self.freq_unit.get() == "hz":
                self.cutoff_freq_str.set("200")
            else:
                self.cutoff_freq_str.set("0.20")
            return
        
        wc2 = None
        if self.filter_type.get() in ["passa-faixa", "rejeita-faixa"]:
            wc2 = self.validate_freq(self.cutoff_freq2_str.get())
            if wc2 is None:
                if self.freq_unit.get() == "hz":
                    self.cutoff_freq2_str.set("400")
                else:
                    self.cutoff_freq2_str.set("0.40")
                return
            if wc1 >= wc2:
                messagebox.showerror("Erro de Frequência", "fc1/ωc1 deve ser menor que fc2/ωc2 para filtros passa-faixa/rejeita-faixa.")
                return
        
        # Formatar valores conforme unidade
        if self.freq_unit.get() == "hz":
            self.cutoff_freq_str.set(f"{wc1:.1f}")
            if wc2 is not None:
                self.cutoff_freq2_str.set(f"{wc2:.1f}")
        else:
            self.cutoff_freq_str.set(f"{wc1:.2f}")
            if wc2 is not None:
                self.cutoff_freq2_str.set(f"{wc2:.2f}")

        self.update_filter()

    def validate_order_and_update(self):
        """Valida a ordem e atualiza o filtro."""
        order_val = self.validate_order(self.order_entry.get())
        if order_val is not None:
            self.filter_order.set(order_val)
            self.order_entry.delete(0, tk.END)
            self.order_entry.insert(0, str(order_val))
            self.update_filter()
        else:
            self.filter_order.set(51)
            self.order_entry.delete(0, tk.END)
            self.order_entry.insert(0, "51")

    def toggle_cutoff2_visibility(self):
         """Mostra ou esconde os controles da segunda frequência de corte."""
         if self.filter_type.get() in ["passa-faixa", "rejeita-faixa"]:
            self.cutoff2_label_widget.grid()
            self.freq2_frame.grid()
         else:
            self.cutoff2_label_widget.grid_remove()
            self.freq2_frame.grid_remove()

    def on_filter_type_change(self, filter_type):
        """
        Atualiza a interface com base no tipo de filtro selecionado
        
        Args:
            filter_type: Tipo de filtro selecionado
        """
        self.toggle_cutoff2_visibility()
        self.validate_and_update()
    
    def get_window(self):
        """
        Obtém a função de janelamento selecionada - IMPLEMENTAÇÃO CONFORME CAPÍTULO 7.5
        
        Returns:
            ndarray: Array com os coeficientes da janela
        """
        N = self.filter_order.get()
        window_name = self.window_type.get()
        
        # Implementação conforme Equação 7.60 do livro (Oppenheim & Schafer)
        # IMPORTANTE: O livro define janelas para 0 ≤ n ≤ M, onde M = N-1
        M = N - 1
        n = np.arange(N)
        
        if window_name == "retangular":
            # Equação 7.60a: w[n] = 1, 0 ≤ n ≤ M
            return np.ones(N)
        
        elif window_name == "hamming":
            # Equação 7.60d: w[n] = 0.54 - 0.46*cos(2πn/M), 0 ≤ n ≤ M
            return 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
        
        elif window_name == "hanning":
            # Equação 7.60c: w[n] = 0.5 - 0.5*cos(2πn/M), 0 ≤ n ≤ M
            return 0.5 - 0.5 * np.cos(2 * np.pi * n / M)
        
        elif window_name == "blackman":
            # Equação 7.60e: w[n] = 0.42 - 0.5*cos(2πn/M) + 0.08*cos(4πn/M), 0 ≤ n ≤ M
            return (0.42 - 0.5 * np.cos(2 * np.pi * n / M) + 
                   0.08 * np.cos(4 * np.pi * n / M))
        
        elif window_name == "bartlett":
            # Equação 7.60b: Janela triangular (Bartlett)
            window = np.zeros(N)
            for i in range(N):
                if i <= M / 2:
                    window[i] = 2 * i / M
                else:
                    window[i] = 2 - 2 * i / M
            return window
        
        elif window_name == "kaiser":
            # Equação 7.72: Janela Kaiser com β configurável
            try:
                beta = float(self.beta_var.get()) if hasattr(self, 'beta_var') else 8.0
                return signal.get_window(("kaiser", beta), N)
            except Exception:
                # Fallback para Hamming
                return 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
        
        else:
            # Fallback para Hamming
            messagebox.showwarning("Janela Inválida", f"Janela '{window_name}' não reconhecida. Usando Hamming.", parent=self.root)
            self.window_type.set("hamming")
            return 0.54 - 0.46 * np.cos(2 * np.pi * n / M)
    
    def ideal_filter(self):
        """
        Calcula a resposta ao impulso do filtro ideal (sem janelamento)
        Implementação conforme Equação 7.71 e seções correspondentes
        
        Returns:
            tuple: (h_ideal, descrição do filtro)
        """
        N = self.filter_order.get()
        M = N - 1  # Conforme notação do livro: M é a ordem, N = M+1 é o comprimento
        filter_type = self.filter_type.get()
        
        # Converter frequências para forma normalizada (0 a 1) para cálculos internos
        wc1 = self.freq_to_normalized(float(self.cutoff_freq_str.get())) * np.pi
        wc2 = None
        if filter_type in ["passa-faixa", "rejeita-faixa"]:
            wc2 = self.freq_to_normalized(float(self.cutoff_freq2_str.get())) * np.pi
        
        # Índices centrados em M/2 para fase linear (Equação 7.71)
        alpha = M / 2
        n = np.arange(N)
        h_ideal = np.zeros(N)
        
        if filter_type == "passa-baixa":
            # Equação 7.70: h[n] = sen[ωc(n-M/2)] / [π(n-M/2)]
            for i in range(N):
                if abs(n[i] - alpha) < 1e-10:  # Evitar divisão por zero
                    h_ideal[i] = wc1 / np.pi
                else:
                    h_ideal[i] = np.sin(wc1 * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
            description = f"Filtro Passa-Baixa (Eq. 7.71)\nFreq. Corte: {wc1/np.pi:.3f}π"
            
        elif filter_type == "passa-alta":
            # Equação 7.80: hhp[n] = δ[n-M/2] - hlp[n]
            for i in range(N):
                if abs(n[i] - alpha) < 1e-10:
                    h_ideal[i] = 1.0 - wc1 / np.pi
                else:
                    # sinc(n-M/2) - hlp[n]
                    sinc_term = np.sin(np.pi * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
                    lowpass_term = np.sin(wc1 * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
                    h_ideal[i] = sinc_term - lowpass_term
            description = f"Filtro Passa-Alta (Eq. 7.80)\nFreq. Corte: {wc1/np.pi:.3f}π"
            
        elif filter_type == "passa-faixa":
            # Diferença de dois passa-baixa
            for i in range(N):
                if abs(n[i] - alpha) < 1e-10:
                    h_ideal[i] = (wc2 - wc1) / np.pi
                else:
                    term1 = np.sin(wc2 * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
                    term2 = np.sin(wc1 * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
                    h_ideal[i] = term1 - term2
            description = (f"Filtro Passa-Faixa\nωc1: {wc1/np.pi:.3f}π\n"
                          f"ωc2: {wc2/np.pi:.3f}π")
            
        elif filter_type == "rejeita-faixa":
            # Impulso menos passa-faixa
            for i in range(N):
                if abs(n[i] - alpha) < 1e-10:
                    h_ideal[i] = 1.0 - (wc2 - wc1) / np.pi
                else:
                    # Impulso centrado
                    sinc_term = np.sin(np.pi * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
                    # Termo passa-faixa
                    term1 = np.sin(wc2 * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
                    term2 = np.sin(wc1 * (n[i] - alpha)) / (np.pi * (n[i] - alpha))
                    bandpass_term = term1 - term2
                    h_ideal[i] = sinc_term - bandpass_term
            description = (f"Filtro Rejeita-Faixa\nωc1: {wc1/np.pi:.3f}π\n"
                          f"ωc2: {wc2/np.pi:.3f}π")
        
        return h_ideal, description
    
    def update_filter(self):
        """Atualiza todos os cálculos e visualizações do filtro"""
        try:
            # Obter a janela
            window = self.get_window()
            
            # Obter o filtro ideal
            h_ideal, description = self.ideal_filter()
            
            # Aplicar janelamento
            h_windowed = h_ideal * window
            
            # Calcular a resposta em frequência
            w, H_ideal = signal.freqz(h_ideal, worN=8000)
            w, H_windowed = signal.freqz(h_windowed, worN=8000)
            
            # Normalizar frequência para unidades de π
            w_normalized = w / np.pi
            
            # Atualizar visualizações
            self.plot_window(window)
            self.plot_coefficients(h_ideal, h_windowed)
            self.plot_frequency_response(w_normalized, H_ideal, H_windowed)
            
            # Atualizar informações e métricas
            self.update_info(description, h_windowed, w_normalized, H_windowed)
        except Exception as e:
            messagebox.showerror("Erro de Cálculo", f"Ocorreu um erro ao atualizar o filtro: {e}", parent=self.root)
    
    def plot_window(self, window):
        """
        Plota a função de janelamento (discreta)
        
        Args:
            window: Array com os coeficientes da janela
        """
        self.window_fig.clear()
        ax = self.window_fig.add_subplot(111)
        
        n = np.arange(len(window))
        markerline, stemlines, baseline = ax.stem(n, window, linefmt="b-", markerfmt="bo", basefmt="k-")
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=1)
        plt.setp(baseline, linewidth=1)
        
        ax.set_title(f"Função de Janelamento: {self.window_type.get().title()} (N={len(window)})")
        ax.set_xlabel("Amostra (n)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        
        self.window_fig.tight_layout()
        self.window_canvas.draw()
    
    def plot_coefficients(self, h_ideal, h_windowed):
        """
        Plota os coeficientes do filtro ideal e janelado
        
        Args:
            h_ideal: Coeficientes do filtro ideal
            h_windowed: Coeficientes do filtro janelado
        """
        self.coef_fig.clear()
        ax = self.coef_fig.add_subplot(111)
        
        N = len(h_ideal)
        M = N - 1
        n = np.arange(N) - M//2  # Centrar em torno de zero
        
        markerline, stemlines, baseline = ax.stem(n, h_windowed, linefmt="b-", markerfmt="bo", basefmt="k-", label="Janelado h[n]")
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=1)
        plt.setp(baseline, linewidth=1)
        
        ax.set_title("Coeficientes do Filtro Janelado")
        ax.set_xlabel("n")
        ax.set_ylabel("h[n]")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.coef_fig.tight_layout()
        self.coef_canvas.draw()
    
    def plot_frequency_response(self, w, H_ideal, H_windowed):
        """
        Plota a resposta em frequência do filtro
        
        Args:
            w: Vetor de frequências normalizadas (0 a 1 para 0 a pi)
            H_ideal: Resposta em frequência do filtro ideal
            H_windowed: Resposta em frequência do filtro janelado
        """
        self.freq_fig.clear()
        
        # Magnitude em dB
        ax1 = self.freq_fig.add_subplot(211)
        H_ideal_db = 20 * np.log10(np.abs(H_ideal) + 1e-10)
        H_windowed_db = 20 * np.log10(np.abs(H_windowed) + 1e-10)
        
        ax1.plot(w, H_ideal_db, "r--", linewidth=1.5, label="Ideal", alpha=0.8)
        ax1.plot(w, H_windowed_db, "b-", linewidth=2, label="Janelado")
        ax1.set_title("Resposta em Frequência")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-100, 5)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Fase
        ax2 = self.freq_fig.add_subplot(212)
        phase_ideal = np.unwrap(np.angle(H_ideal))
        phase_windowed = np.unwrap(np.angle(H_windowed))
        
        if self.show_compensated_phase.get():
            # Remover o atraso linear da fase para melhor visualização
            N = self.filter_order.get()
            M = N - 1
            delay = M / 2
            phase_ideal_plot = phase_ideal + delay * w * np.pi
            phase_windowed_plot = phase_windowed + delay * w * np.pi
            ylabel = "Fase Compensada (rad)"
            title_suffix = " (Compensada)"
        else:
            # Mostrar fase original
            phase_ideal_plot = phase_ideal
            phase_windowed_plot = phase_windowed
            ylabel = "Fase Original (rad)"
            title_suffix = " (Original)"
        
        ax2.plot(w, phase_ideal_plot, "r--", linewidth=1.5, label=f"Ideal{title_suffix}", alpha=0.8)
        ax2.plot(w, phase_windowed_plot, "b-", linewidth=2, label=f"Janelado{title_suffix}")
        ax2.set_xlabel("Frequência Normalizada (× π rad/amostra)")
        ax2.set_ylabel(ylabel)
        ax2.set_xlim(0, 1)
        if self.show_compensated_phase.get():
            ax2.set_ylim(-np.pi, np.pi)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        self.freq_fig.tight_layout()
        self.freq_canvas.draw()
    
    def find_freq_at_db(self, w, H_db, target_db):
        """Encontra a primeira frequência onde a magnitude atinge target_db."""
        try:
            indices = np.where(np.diff(np.sign(H_db - target_db)))[0]
            if len(indices) > 0:
                idx = indices[0]
                if idx + 1 < len(w):
                    w1, w2 = w[idx], w[idx+1]
                    db1, db2 = H_db[idx], H_db[idx+1]
                    if abs(db2 - db1) > 1e-6:
                        freq = w1 + (target_db - db1) * (w2 - w1) / (db2 - db1)
                        return freq
                    else:
                        return w1
            return None
        except Exception:
            return None

    def update_info(self, description, h_windowed, w, H_windowed):
        """
        Atualiza as informações sobre o filtro, incluindo métricas detalhadas.
        Implementação baseada nas análises do Capítulo 7.5
        
        Args:
            description: Descrição básica do filtro.
            h_windowed: Coeficientes do filtro janelado.
            w: Vetor de frequências normalizadas.
            H_windowed: Resposta em frequência do filtro janelado.
        """
        H_db = 20 * np.log10(np.abs(H_windowed) + 1e-10)
        N = len(h_windowed)
        M = N - 1  # Ordem conforme notação do livro
        
        # Cálculo das Métricas conforme Tabela 7.2 e Figura 7.31
        passband_edge = None
        stopband_edge = None
        transition_width = None
        min_stopband_atten_db = 0

        # Frequências de corte nominais (sempre normalizadas para cálculos internos)
        wc1_norm = self.freq_to_normalized(float(self.cutoff_freq_str.get()))
        
        # Encontrar borda da banda passante (-3dB)
        passband_edge = self.find_freq_at_db(w, H_db, -3.0)
        
        # Encontrar borda da banda de rejeição
        target_stop_db = -40.0 
        if wc1_norm < 0.8:
            stop_region = w > (wc1_norm + 0.05)
            if np.any(stop_region):
                stopband_edge = self.find_freq_at_db(w[stop_region], H_db[stop_region], target_stop_db)
        
        if stopband_edge is None:
             target_stop_db = -30.0
             if wc1_norm < 0.8:
                 stop_region = w > (wc1_norm + 0.05)
                 if np.any(stop_region):
                     stopband_edge = self.find_freq_at_db(w[stop_region], H_db[stop_region], target_stop_db)
        
        # Calcular largura da banda de transição
        if passband_edge is not None and stopband_edge is not None:
            transition_width = abs(stopband_edge - passband_edge)
        
        # Atenuação mínima na banda de rejeição
        if wc1_norm < 0.9:
            stop_indices = w > (wc1_norm + 0.1)
            if np.any(stop_indices):
                min_stopband_atten_db = -np.max(H_db[stop_indices])
            else:
                min_stopband_atten_db = -np.min(H_db)
        else:
            min_stopband_atten_db = -np.min(H_db)

        # Informações teóricas da janela (Tabela 7.2)
        window_info = self.get_window_info()
        
        # Largura teórica do lóbulo principal
        window_name = self.window_type.get()
        if window_name == "retangular":
            theoretical_width = f"4π/{N} = {4/N:.3f}π"
        elif window_name in ["hamming", "hanning"]:
            theoretical_width = f"8π/{M} = {8/M:.3f}π"
        elif window_name == "blackman":
            theoretical_width = f"12π/{M} = {12/M:.3f}π"
        elif window_name == "bartlett":
            theoretical_width = f"8π/{M} = {8/M:.3f}π"
        else:
            theoretical_width = "Variável"

        # Verificação de simetria e tipo de fase linear
        is_symmetric = np.allclose(h_windowed, h_windowed[::-1], atol=1e-10)
        
        if N % 2 == 1:  # N ímpar
            if is_symmetric:
                filter_type_linear = "Tipo I (fase linear, M par)"
            else:
                filter_type_linear = "Tipo III (antissimétrica, M par)"
        else:  # N par  
            if is_symmetric:
                filter_type_linear = "Tipo II (fase linear, M ímpar)"
            else:
                filter_type_linear = "Tipo IV (antissimétrica, M ímpar)"

        # Montar Texto de Informações
        info = (
            f"{description}\n"
            f"{'='*40}\n\n"
            f"PARÂMETROS DO PROJETO:\n"
            f"• Janela: {window_name.title()}\n"
            f"• Ordem (M): {M}\n"
            f"• Comprimento (N=M+1): {N}\n"
            f"• Largura Teórica: {theoretical_width}\n"
        )
        
        # Adicionar informação sobre frequência de amostragem se em Hz
        if self.freq_unit.get() == "hz":
            fs = float(self.fs_var.get())
            info += f"• Freq. Amostragem: {fs:.0f} Hz\n"
            info += f"• Freq. Nyquist: {fs/2:.0f} Hz\n"
        
        info += f"\nCARACTERÍSTICAS DA JANELA:\n• {window_info}\n\nMÉTRICAS MEDIDAS:\n"
        
        if passband_edge:
            info += f"• Borda Passante (-3dB): {passband_edge:.3f}π\n"
        else:
            info += "• Borda Passante (-3dB): N/A\n"
            
        if stopband_edge:
            info += f"• Borda Rejeição ({target_stop_db:.0f}dB): {stopband_edge:.3f}π\n"
        else:
            info += f"• Borda Rejeição ({target_stop_db:.0f}dB): N/A\n"
            
        if transition_width:
            info += f"• Largura Transição (Δω): {transition_width:.3f}π\n"
        else:
            info += "• Largura Transição: N/A\n"
            
        info += f"• Atenuação Mín.: {min_stopband_atten_db:.1f} dB\n\n"
        
        info += (
            f"ANÁLISE ESTRUTURAL:\n"
            f"• Simetria: {'Sim' if is_symmetric else 'Não'}\n"
            f"• Tipo: {filter_type_linear}\n"
            f"• Estabilidade: Garantida (FIR)\n"
            f"• Fase: Linear generalizada\n"
            f"• Atraso: {M/2:.1f} amostras\n\n"
            f"OBSERVAÇÕES:\n"
            f"• Conforme Oppenheim & Schafer Cap. 7.5\n"
            f"• Fenômeno de Gibbs controlado\n"
            f"• Compromisso: transição ↔ supressão\n"
        )
        
        # Atualizar o widget de texto
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, info)
        self.info_text.config(state=tk.DISABLED)

    def get_window_info(self):
        """Retorna informações específicas sobre a janela selecionada conforme Tabela 7.2."""
        window_name = self.window_type.get()
        
        # Informações baseadas na Tabela 7.2 do Oppenheim & Schafer
        window_descriptions = {
            "retangular": "Lóbulo lateral: -13dB, Transição estreita",
            "hamming": "Lóbulo lateral: -41dB, Erro pico: -53dB",
            "hanning": "Lóbulo lateral: -31dB, Erro pico: -44dB", 
            "blackman": "Lóbulo lateral: -57dB, Erro pico: -74dB",
            "bartlett": "Lóbulo lateral: -25dB, Triangular",
            "kaiser": f"Paramétrica, β={self.beta_var.get() if hasattr(self, 'beta_var') else '8.0'}, ajustável"
        }
        
        return window_descriptions.get(window_name, "Janela personalizada")

def main():
    """Função principal para iniciar o aplicativo"""
    root = tk.Tk()
    app = FilterDesignApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()