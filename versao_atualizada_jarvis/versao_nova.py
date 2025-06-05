"""
Software de Projeto de Filtros FIR
Desenvolvido seguindo a metodologia do Problema 03
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
    Aplicativo para projeto de filtros FIR com janelamento
    """
    
    def __init__(self, root):
        """Inicializa a aplicação"""
        self.root = root
        self.root.title("Projeto de Filtros FIR - Janelamento da Função Seno Cardinal")
        self.root.geometry("1600x900")
        self.root.minsize(1400, 800)
        
        # Configuração de estilo
        self.style = ttk.Style()
        self.style.theme_use("clam")
        
        # Parâmetros das janelas (da tabela fornecida)
        self.window_parameters = {
            'Retangular': {
                'largura_transicao_normalizada': 0.9,
                'ondulacao_banda_passante_db': 0.7416,
                'lobulo_principal_lateral_db': 13,
                'atenuacao_banda_rejeicao_db': 21,
                'expressao': "w(n) = 1"
            },
            'Bartlett': {
                'largura_transicao_normalizada': 2.3,
                'ondulacao_banda_passante_db': 0.185,
                'lobulo_principal_lateral_db': 25,
                'atenuacao_banda_rejeicao_db': 25,
                'expressao': "w(n) = 2 - 2n/M"
            },
            'Hanning': {
                'largura_transicao_normalizada': 3.1,
                'ondulacao_banda_passante_db': 0.0546,
                'lobulo_principal_lateral_db': 31,
                'atenuacao_banda_rejeicao_db': 44,
                'expressao': "w(n) = 0.5 + 0.5*cos(2πn/N)"
            },
            'Hamming': {
                'largura_transicao_normalizada': 3.3,
                'ondulacao_banda_passante_db': 0.0194,
                'lobulo_principal_lateral_db': 41,
                'atenuacao_banda_rejeicao_db': 53,
                'expressao': "w(n) = 0.54 + 0.46*cos(2πn/N)"
            },
            'Blackman': {
                'largura_transicao_normalizada': 5.5,
                'ondulacao_banda_passante_db': 0.0017,
                'lobulo_principal_lateral_db': 57,
                'atenuacao_banda_rejeicao_db': 75,
                'expressao': "w(n) = 0.42 + 0.5*cos(2πn/N) + 0.08*cos(4πn/(N-1))"
            },
            'Kaiser_beta_4.54': {
                'largura_transicao_normalizada': 2.93,
                'ondulacao_banda_passante_db': 0.0274,
                'lobulo_principal_lateral_db': None,
                'atenuacao_banda_rejeicao_db': 50,
                'beta': 4.54,
                'expressao': "I₀(β√(1-(2n/(N-1))²))/I₀(β)"
            },
            'Kaiser_beta_6.76': {
                'largura_transicao_normalizada': 4.32,
                'ondulacao_banda_passante_db': 0.00275,
                'lobulo_principal_lateral_db': None,
                'atenuacao_banda_rejeicao_db': 70,
                'beta': 6.76,
                'expressao': "I₀(β√(1-(2n/(N-1))²))/I₀(β)"
            },
            'Kaiser_beta_8.96': {
                'largura_transicao_normalizada': 5.71,
                'ondulacao_banda_passante_db': 0.000275,
                'lobulo_principal_lateral_db': None,
                'atenuacao_banda_rejeicao_db': 90,
                'beta': 8.96,
                'expressao': "I₀(β√(1-(2n/(N-1))²))/I₀(β)"
            }
        }
        
        # Variáveis de entrada
        self.filter_type_var = tk.StringVar(value="Passa-Baixa")
        self.fs_var = tk.StringVar(value="8000")  # Frequência de amostragem
        self.fp_var = tk.StringVar(value="1500")  # Frequência da borda da banda passante
        self.transition_width_var = tk.StringVar(value="500")  # Largura de transição
        self.stopband_atten_var = tk.StringVar(value="50")  # Atenuação na banda de rejeição
        
        # Variável para janela selecionada
        self.selected_window_var = tk.StringVar()
        
        # Janelas disponíveis (será filtrada baseada na atenuação)
        self.available_windows = []
        
        # Resultados
        self.filter_coeffs = None
        self.calculated_order = None
        self.window_function = None
        self.ideal_response = None
        
        self.create_interface()
        
    def create_interface(self):
        """Cria a interface completa"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame de entrada (esquerda)
        input_frame = ttk.LabelFrame(main_frame, text="Especificações e Controles", padding=15)
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        input_frame.configure(width=400)
        
        # Frame de visualização (direita)
        viz_frame = ttk.LabelFrame(main_frame, text="Visualização dos Resultados", padding=10)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_input_section(input_frame)
        self.create_visualization_section(viz_frame)
    
    def create_input_section(self, parent):
        """Cria a seção de entrada de dados"""
        row = 0
        
        # Título
        title_label = ttk.Label(parent, text="PROJETO DE FILTROS FIR", 
                               font=('Arial', 12, 'bold'))
        title_label.grid(row=row, column=0, columnspan=2, pady=(0, 20))
        row += 1
        
        # Tipo de filtro (dropdown esquerdo)
        ttk.Label(parent, text="Tipo de Filtro:", font=('Arial', 10, 'bold')).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        filter_combo = ttk.Combobox(parent, textvariable=self.filter_type_var, 
                                   values=["Passa-Baixa", "Passa-Alta"],
                                   state="readonly", width=20)
        filter_combo.grid(row=row, column=1, sticky=tk.W+tk.E, pady=5, padx=(10,0))
        row += 1
        
        # Separador
        ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, 
                                                       sticky="ew", pady=15)
        row += 1
        
        # Título das especificações
        spec_title = ttk.Label(parent, text="ESPECIFICAÇÕES DO FILTRO", 
                              font=('Arial', 11, 'bold'))
        spec_title.grid(row=row, column=0, columnspan=2, pady=(0, 10))
        row += 1
        
        # Frequência de amostragem
        ttk.Label(parent, text="Frequência de Amostragem:", font=('Arial', 10)).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        fs_frame = ttk.Frame(parent)
        fs_frame.grid(row=row, column=1, sticky=tk.W+tk.E, pady=5, padx=(10,0))
        ttk.Entry(fs_frame, textvariable=self.fs_var, width=10).pack(side=tk.LEFT)
        ttk.Label(fs_frame, text="Hz").pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        
        # Frequência da borda da banda passante
        ttk.Label(parent, text="Frequência da Borda da\nBanda Passante:", font=('Arial', 10)).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        fp_frame = ttk.Frame(parent)
        fp_frame.grid(row=row, column=1, sticky=tk.W+tk.E, pady=5, padx=(10,0))
        ttk.Entry(fp_frame, textvariable=self.fp_var, width=10).pack(side=tk.LEFT)
        ttk.Label(fp_frame, text="Hz").pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        
        # Largura de transição
        ttk.Label(parent, text="Largura de Transição:", font=('Arial', 10)).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        tw_frame = ttk.Frame(parent)
        tw_frame.grid(row=row, column=1, sticky=tk.W+tk.E, pady=5, padx=(10,0))
        ttk.Entry(tw_frame, textvariable=self.transition_width_var, width=10).pack(side=tk.LEFT)
        ttk.Label(tw_frame, text="Hz").pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        
        # Atenuação na banda de rejeição
        ttk.Label(parent, text="Atenuação na Banda\nde Rejeição:", font=('Arial', 10)).grid(
            row=row, column=0, sticky=tk.W, pady=5)
        atten_frame = ttk.Frame(parent)
        atten_frame.grid(row=row, column=1, sticky=tk.W+tk.E, pady=5, padx=(10,0))
        ttk.Entry(atten_frame, textvariable=self.stopband_atten_var, width=10).pack(side=tk.LEFT)
        ttk.Label(atten_frame, text="dB").pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        
        # Botão para verificar janelas disponíveis
        check_button = ttk.Button(parent, text="Verificar Janelas Disponíveis", 
                                 command=self.check_available_windows)
        check_button.grid(row=row, column=0, columnspan=2, pady=20)
        row += 1
        
        # Frame para seleção de janelas
        self.window_frame = ttk.LabelFrame(parent, text="Funções de Janelamento Disponíveis", 
                                          padding=10)
        self.window_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        row += 1
        
        # Botão para calcular filtro
        self.calc_button = ttk.Button(parent, text="PROJETAR FILTRO", 
                                     command=self.design_filter,
                                     state=tk.DISABLED)
        self.calc_button.grid(row=row, column=0, columnspan=2, pady=20)
        row += 1
        
        # Frame de resultados calculados
        self.results_frame = ttk.LabelFrame(parent, text="Informações Calculadas", padding=10)
        self.results_frame.grid(row=row, column=0, columnspan=2, sticky="nsew", pady=10)
        
        self.results_text = tk.Text(self.results_frame, height=15, width=50, wrap=tk.WORD, 
                                   font=('Consolas', 9))
        scrollbar_results = ttk.Scrollbar(self.results_frame, orient="vertical", 
                                         command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar_results.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_results.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Configurar expansão
        parent.grid_rowconfigure(row, weight=1)
        parent.grid_columnconfigure(1, weight=1)
    
    def create_visualization_section(self, parent):
        """Cria a seção de visualização"""
        # Notebook para múltiplas visualizações
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Aba 1: Função de janelamento
        self.window_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.window_tab, text="Função de Janelamento")
        
        # Aba 2: Coeficientes do filtro
        self.coef_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.coef_tab, text="Coeficientes do Filtro")
        
        # Aba 3: Resposta em frequência
        self.freq_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.freq_tab, text="Resposta em Frequência")
        
        self.setup_plots()
    
    def setup_plots(self):
        """Configura os gráficos"""
        # Gráfico da função de janelamento
        self.window_fig = Figure(figsize=(12, 6), dpi=100)
        self.window_canvas = FigureCanvasTkAgg(self.window_fig, master=self.window_tab)
        self.window_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.window_canvas, self.window_tab)
        
        # Gráfico de coeficientes
        self.coef_fig = Figure(figsize=(12, 8), dpi=100)
        self.coef_canvas = FigureCanvasTkAgg(self.coef_fig, master=self.coef_tab)
        self.coef_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.coef_canvas, self.coef_tab)
        
        # Gráfico de resposta em frequência
        self.freq_fig = Figure(figsize=(12, 8), dpi=100)
        self.freq_canvas = FigureCanvasTkAgg(self.freq_fig, master=self.freq_tab)
        self.freq_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.freq_canvas, self.freq_tab)
    
    def check_available_windows(self):
        """Verifica quais janelas atendem a especificação de atenuação"""
        try:
            required_atten = float(self.stopband_atten_var.get())
            
            # Limpar frame anterior
            for widget in self.window_frame.winfo_children():
                widget.destroy()
            
            self.available_windows = []
            
            # Verificar cada janela
            for window_name, params in self.window_parameters.items():
                if params['atenuacao_banda_rejeicao_db'] >= required_atten:
                    self.available_windows.append(window_name)
            
            if not self.available_windows:
                messagebox.showwarning("Aviso", 
                    f"Nenhuma janela disponível atende a especificação de {required_atten} dB.\n"
                    "Considere reduzir a exigência de atenuação.")
                return
            
            # Criar dropdown com janelas disponíveis
            ttk.Label(self.window_frame, 
                     text=f"Janelas que atendem ≥{required_atten} dB:", 
                     font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 10))
            
            window_combo = ttk.Combobox(self.window_frame, textvariable=self.selected_window_var,
                                       values=self.available_windows, state="readonly", width=25)
            window_combo.pack(pady=5)
            window_combo.bind("<<ComboboxSelected>>", self.on_window_selected)
            
            # Mostrar informações das janelas disponíveis
            info_text = "\n=== JANELAS DISPONÍVEIS ===\n"
            for window_name in self.available_windows:
                params = self.window_parameters[window_name]
                info_text += f"\n{window_name}:\n"
                info_text += f"• Atenuação: {params['atenuacao_banda_rejeicao_db']} dB\n"
                info_text += f"• Largura de Transição: {params['largura_transicao_normalizada']}/N\n"
                info_text += f"• Ondulação Banda Passante: {params['ondulacao_banda_passante_db']} dB\n"
                if params.get('lobulo_principal_lateral_db'):
                    info_text += f"• Lóbulo Principal vs Lateral: {params['lobulo_principal_lateral_db']} dB\n"
                info_text += f"• Expressão: {params['expressao']}\n"
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, info_text)
            
        except ValueError:
            messagebox.showerror("Erro", "Digite um valor válido para a atenuação.")
    
    def on_window_selected(self, event=None):
        """Callback quando uma janela é selecionada"""
        if self.selected_window_var.get():
            self.calc_button.config(state=tk.NORMAL)
    
    def design_filter(self):
        """Projeta o filtro com as especificações fornecidas"""
        try:
            # Obter especificações
            fs = float(self.fs_var.get())
            fp = float(self.fp_var.get())
            transition_width = float(self.transition_width_var.get())
            stopband_atten = float(self.stopband_atten_var.get())
            filter_type = self.filter_type_var.get()
            window_name = self.selected_window_var.get()
            
            # Validações
            if fp >= fs/2:
                raise ValueError("Frequência da banda passante deve ser menor que Fs/2")
            
            # Calcular frequência de corte baseada no tipo de filtro
            if filter_type == "Passa-Baixa":
                fs_freq = fp + transition_width  # Frequência de stopband
                fc = (fp + fs_freq) / 2  # Frequência de corte centrada na transição
            else:  # Passa-Alta
                fs_freq = fp - transition_width  # Frequência de stopband
                fc = (fs_freq + fp) / 2  # Frequência de corte centrada na transição
                
                if fs_freq <= 0:
                    raise ValueError("Para passa-alta: fp - largura_transição deve ser > 0")
            
            # Calcular ordem do filtro
            delta_f_norm = transition_width / fs
            window_params = self.window_parameters[window_name]
            
            # Usar fator da janela para calcular ordem
            if 'Kaiser' in window_name:
                # Para Kaiser, usar fórmula específica
                A = stopband_atten
                beta = window_params['beta']
                M = (A - 8) / (2.285 * 2 * np.pi * delta_f_norm)
                order = int(np.ceil(M))
            else:
                # Para outras janelas, usar fator de largura de transição
                factor = window_params['largura_transicao_normalizada']
                N_calc = factor / delta_f_norm
                order = int(np.ceil(N_calc))
            
            # Garantir ordem ímpar para simetria
            if order % 2 == 0:
                order += 1
            
            # Limitar ordem
            order = max(11, min(501, order))
            
            # Projetar filtro ideal
            nyquist = fs / 2
            fc_norm = fc / nyquist
            
            if filter_type == "Passa-Baixa":
                h_ideal = self.ideal_lowpass(order, fc_norm)
            else:  # Passa-Alta
                h_ideal = self.ideal_highpass(order, fc_norm)
            
            # Criar janela
            if 'Kaiser' in window_name:
                beta = window_params['beta']
                window = signal.get_window(('kaiser', beta), order)
            elif window_name == 'Retangular':
                window = signal.get_window('boxcar', order)
            elif window_name == 'Bartlett':
                window = signal.get_window('bartlett', order)
            elif window_name == 'Hanning':
                window = signal.get_window('hann', order)
            elif window_name == 'Hamming':
                window = signal.get_window('hamming', order)
            elif window_name == 'Blackman':
                window = signal.get_window('blackman', order)
            
            # Aplicar janelamento
            h_windowed = h_ideal * window
            
            # Armazenar resultados
            self.filter_coeffs = h_windowed
            self.calculated_order = order
            self.window_function = window
            self.ideal_response = h_ideal
            
            # Atualizar visualizações
            self.update_all_plots(h_windowed, window, h_ideal, fs, fc)
            
            # Mostrar cálculos detalhados
            self.show_calculations(fs, fp, transition_width, stopband_atten, 
                                 filter_type, window_name, order, fc, 
                                 h_windowed, delta_f_norm)
            
            messagebox.showinfo("Sucesso", "Filtro projetado com sucesso!")
            
        except ValueError as e:
            messagebox.showerror("Erro de Entrada", str(e))
        except Exception as e:
            messagebox.showerror("Erro", f"Erro no projeto: {e}")
    
    def ideal_lowpass(self, N, fc_norm):
        """Calcula filtro passa-baixa ideal"""
        M = N - 1
        n = np.arange(N)
        h = np.zeros(N)
        
        for i in range(N):
            n_centered = n[i] - M/2
            
            if abs(n_centered) < 1e-10:
                h[i] = 2 * fc_norm
            else:
                omega_c = fc_norm * np.pi
                h[i] = 2 * fc_norm * np.sin(omega_c * n_centered) / (omega_c * n_centered)
        
        return h
    
    def ideal_highpass(self, N, fc_norm):
        """Calcula filtro passa-alta ideal"""
        M = N - 1
        n = np.arange(N)
        h = np.zeros(N)
        
        for i in range(N):
            n_centered = n[i] - M/2
            
            if abs(n_centered) < 1e-10:
                h[i] = 1.0 - 2 * fc_norm
            else:
                # Impulso delta menos passa-baixa
                delta_impulse = np.sin(np.pi * n_centered) / (np.pi * n_centered)
                omega_c = fc_norm * np.pi
                lowpass_term = 2 * fc_norm * np.sin(omega_c * n_centered) / (omega_c * n_centered)
                h[i] = delta_impulse - lowpass_term
        
        return h
    
    def show_calculations(self, fs, fp, transition_width, stopband_atten, 
                         filter_type, window_name, order, fc, h_windowed, delta_f_norm):
        """Mostra os cálculos detalhados"""
        
        calculations = f"""PROJETO DE FILTRO FIR - RESULTADOS DETALHADOS
{'='*60}

ESPECIFICAÇÕES FORNECIDAS:
• Tipo de Filtro: {filter_type}
• Frequência de Amostragem: {fs:.0f} Hz
• Frequência da Borda da Banda Passante: {fp:.0f} Hz
• Largura de Transição: {transition_width:.0f} Hz
• Atenuação na Banda de Rejeição: ≥{stopband_atten} dB
• Janela Selecionada: {window_name}

CÁLCULOS REALIZADOS:
"""
        
        if filter_type == "Passa-Baixa":
            fs_freq = fp + transition_width
            calculations += f"""
• Frequência de Stopband: fs = {fp:.0f} + {transition_width:.0f} = {fs_freq:.0f} Hz
• Frequência de Corte (centrada): fc = ({fp:.0f} + {fs_freq:.0f})/2 = {fc:.0f} Hz"""
        else:
            fs_freq = fp - transition_width
            calculations += f"""
• Frequência de Stopband: fs = {fp:.0f} - {transition_width:.0f} = {fs_freq:.0f} Hz
• Frequência de Corte (centrada): fc = ({fs_freq:.0f} + {fp:.0f})/2 = {fc:.0f} Hz"""
        
        calculations += f"""
• Largura de Transição Normalizada: Δf = {transition_width:.0f}/{fs:.0f} = {delta_f_norm:.6f}

CÁLCULO DA ORDEM DO FILTRO:
• Janela: {window_name}
• Fator da Janela: {self.window_parameters[window_name]['largura_transicao_normalizada']}
• N = {self.window_parameters[window_name]['largura_transicao_normalizada']}/Δf = {self.window_parameters[window_name]['largura_transicao_normalizada']}/{delta_f_norm:.6f} = {self.window_parameters[window_name]['largura_transicao_normalizada']/delta_f_norm:.1f}
• Ordem escolhida (ímpar): N = {order}

CARACTERÍSTICAS DO FILTRO PROJETADO:
• Tipo: FIR Fase Linear Tipo I
• Ordem: M = {order-1}
• Comprimento: N = {order}
• Frequência de Corte Normalizada: fc = {fc/(fs/2):.6f}
• Atraso de Grupo: {(order-1)/2:.1f} amostras
• Simetria: h(n) = h(N-1-n) ✓

PRIMEIROS COEFICIENTES CALCULADOS:"""
        
        # Mostrar alguns coeficientes
        center = (order - 1) // 2
        for i in range(min(6, len(h_windowed))):
            calculations += f"""
h({i}) = {h_windowed[i]:.8f}"""
        
        if len(h_windowed) > 6:
            calculations += f"""
...
h({center}) = {h_windowed[center]:.8f} (centro)
...
h({len(h_windowed)-1}) = {h_windowed[-1]:.8f}"""
        
        calculations += f"""

PROPRIEDADES DA JANELA {window_name.upper()}:
• Atenuação na Banda de Rejeição: {self.window_parameters[window_name]['atenuacao_banda_rejeicao_db']} dB
• Largura de Transição: {self.window_parameters[window_name]['largura_transicao_normalizada']}/N
• Ondulação na Banda Passante: {self.window_parameters[window_name]['ondulacao_banda_passante_db']} dB"""
        
        if self.window_parameters[window_name].get('lobulo_principal_lateral_db'):
            calculations += f"""
• Lóbulo Principal vs Lateral: {self.window_parameters[window_name]['lobulo_principal_lateral_db']} dB"""
        
        calculations += f"""
• Expressão Matemática: {self.window_parameters[window_name]['expressao']}

O filtro resultante deve atender às especificações de desempenho
conforme verificado na resposta em frequência.
"""
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, calculations)
    
    def update_all_plots(self, h_windowed, window, h_ideal, fs, fc):
        """Atualiza todos os gráficos"""
        self.plot_window(window)
        self.plot_coefficients(h_windowed, h_ideal)
        self.plot_frequency_response(h_windowed, fs, fc)
    
    def plot_window(self, window):
        """Plota a função de janelamento"""
        self.window_fig.clear()
        ax = self.window_fig.add_subplot(111)
        
        n = np.arange(len(window))
        ax.plot(n, window, 'b-', linewidth=2, label='Função de Janelamento')
        
        # Stem plot sem alpha
        markerline, stemlines, baseline = ax.stem(n, window, linefmt='b-', markerfmt='bo', basefmt='k-')
        plt.setp(stemlines, alpha=0.7)
        plt.setp(markerline, alpha=0.7)
        
        window_name = self.selected_window_var.get()
        ax.set_title(f'Função de Janelamento: {window_name}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Índice da Amostra (n)')
        ax.set_ylabel('Amplitude w(n)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Destacar centro
        center = len(window) // 2
        ax.axvline(center, color='red', linestyle='--', alpha=0.5, label=f'Centro (n={center})')
        
        self.window_fig.tight_layout()
        self.window_canvas.draw()
    
    def plot_coefficients(self, h_windowed, h_ideal):
        """Plota os coeficientes do filtro"""
        self.coef_fig.clear()
        
        # Subplot 1: Coeficientes ideais vs janelados
        ax1 = self.coef_fig.add_subplot(211)
        n = np.arange(len(h_windowed))
        
        # Coeficientes janelados
        markerline, stemlines, baseline = ax1.stem(n, h_windowed, linefmt='b-', 
                                                  markerfmt='bo', basefmt='k-', 
                                                  label='Coeficientes Janelados h(n)')
        plt.setp(markerline, markersize=4)
        plt.setp(stemlines, linewidth=1.5)
        
        # Coeficientes ideais (linha)
        ax1.plot(n, h_ideal, 'r--', alpha=0.8, linewidth=2, label='Resposta Ideal')
        
        ax1.set_title('Coeficientes do Filtro FIR', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Índice da Amostra (n)')
        ax1.set_ylabel('Amplitude h(n)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Destacar centro
        center = len(h_windowed) // 2
        ax1.axvline(center, color='gray', linestyle=':', alpha=0.5)
        
        # Subplot 2: Zoom na região central
        ax2 = self.coef_fig.add_subplot(212)
        center_range = 10
        start_idx = max(0, center - center_range)
        end_idx = min(len(h_windowed), center + center_range + 1)
        
        n_zoom = n[start_idx:end_idx]
        h_zoom = h_windowed[start_idx:end_idx]
        h_ideal_zoom = h_ideal[start_idx:end_idx]
        
        markerline2, stemlines2, baseline2 = ax2.stem(n_zoom, h_zoom, linefmt='b-', 
                                                     markerfmt='bo', basefmt='k-', 
                                                     label='Coeficientes Janelados')
        plt.setp(markerline2, markersize=5)
        plt.setp(stemlines2, linewidth=2)
        
        ax2.plot(n_zoom, h_ideal_zoom, 'r--', alpha=0.8, linewidth=2, label='Ideal')
        
        ax2.set_title(f'Região Central (n = {start_idx} a {end_idx-1})')
        ax2.set_xlabel('Índice da Amostra (n)')
        ax2.set_ylabel('Amplitude h(n)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.coef_fig.tight_layout()
        self.coef_canvas.draw()
    
    def plot_frequency_response(self, h_windowed, fs, fc):
        """Plota a resposta em frequência"""
        self.freq_fig.clear()
        
        # Calcular resposta em frequência
        w, H = signal.freqz(h_windowed, worN=8192, fs=fs)
        
        # Subplot 1: Magnitude em dB
        ax1 = self.freq_fig.add_subplot(211)
        H_db = 20 * np.log10(np.abs(H) + 1e-10)
        ax1.plot(w, H_db, 'b-', linewidth=2, label='Resposta do Filtro Projetado')
        
        ax1.set_title('Resposta em Magnitude', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Magnitude (dB)')
        ax1.set_ylim(-120, 10)
        ax1.grid(True, alpha=0.3)
        
        # Adicionar linhas de referência
        try:
            fp = float(self.fp_var.get())
            transition_width = float(self.transition_width_var.get())
            stopband_atten = float(self.stopband_atten_var.get())
            filter_type = self.filter_type_var.get()
            
            if filter_type == "Passa-Baixa":
                fs_freq = fp + transition_width
                ax1.axvline(fp, color='g', linestyle='--', alpha=0.8, 
                           label=f'fp = {fp:.0f} Hz')
                ax1.axvline(fs_freq, color='r', linestyle='--', alpha=0.8, 
                           label=f'fs = {fs_freq:.0f} Hz')
            else:  # Passa-Alta
                fs_freq = fp - transition_width
                ax1.axvline(fs_freq, color='r', linestyle='--', alpha=0.8, 
                           label=f'fs = {fs_freq:.0f} Hz')
                ax1.axvline(fp, color='g', linestyle='--', alpha=0.8, 
                           label=f'fp = {fp:.0f} Hz')
            
            ax1.axvline(fc, color='orange', linestyle=':', alpha=0.8, 
                       label=f'fc = {fc:.0f} Hz')
            ax1.axhline(-stopband_atten, color='r', linestyle=':', alpha=0.7, 
                       label=f'Spec: -{stopband_atten:.0f} dB')
            ax1.axhline(-3, color='purple', linestyle=':', alpha=0.7, label='-3 dB')
            
        except:
            pass
        
        ax1.legend(fontsize=9)
        
        # Subplot 2: Fase
        ax2 = self.freq_fig.add_subplot(212)
        phase = np.unwrap(np.angle(H))
        ax2.plot(w, phase, 'b-', linewidth=2)
        ax2.set_xlabel('Frequência (Hz)')
        ax2.set_ylabel('Fase (rad)')
        ax2.set_title('Resposta em Fase')
        ax2.grid(True, alpha=0.3)
        
        # Adicionar linha de referência do atraso de grupo
        if hasattr(self, 'calculated_order') and self.calculated_order:
            group_delay = (self.calculated_order - 1) / 2
            expected_phase = -w * 2 * np.pi * group_delay / fs
            ax2.plot(w, expected_phase, 'r--', alpha=0.6, 
                    label=f'Fase Linear Ideal (Atraso = {group_delay:.1f})')
            ax2.legend()
        
        self.freq_fig.tight_layout()
        self.freq_canvas.draw()

def main():
    """Função principal"""
    root = tk.Tk()
    app = FilterDesignApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()