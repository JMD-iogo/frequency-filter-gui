# Frequency Filter GUI

Aplicação em Python/Tkinter que permite aplicar filtros no domínio da frequência  
(LPF, HPF, BPF e BRF) a imagens em tons de cinzento ou RGB.

---

## Funcionalidades

- Carregar imagens (PNG, JPG, BMP, TIFF).
- Converter automaticamente imagens RGB para intensidade (opcional).
- Visualização:
  - Imagem original
  - Espectro de magnitude (FFT)
  - Fase
  - Máscara de filtro
  - Magnitude filtrada
  - Imagem resultante
- Filtros disponíveis:
  - **LPF** – passa-baixo  
  - **HPF** – passa-alto  
  - **BPF** – passa-banda  
  - **BRF** – rejeita-banda  
- Perfis:
  - Ideal  
  - Gaussiano  
  - Butterworth (com ordem ajustável)
- Modo **notch filter** (filtro deslocado), com seleção do centro por clique.
- Atualização em tempo real (máscara / filtro).
- Cálculo de métricas:
  - MSE  
  - PSNR  
- Guardar automaticamente:
  - imagem filtrada
  - imagem original
  - espectro de magnitude original
  - fase
  - máscara do filtro
  - espectro filtrado
  - ficheiro de especificações

---

## Instalação

```bash
git clone https://github.com/<teu-username>/frequency-filter-gui.git
cd frequency-filter-gui

# (opcional) criar ambiente virtual
python -m venv venv
venv\Scripts\activate

# instalar dependências
pip install -r requirements.txt
