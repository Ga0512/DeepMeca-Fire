# 🔥 **FireDetectAPI — Detecção de Incêndios com YOLOv8**

> Sistema inteligente de **monitoramento de incêndios em tempo real**, desenvolvido como **Trabalho de Conclusão de Curso (TCC)**.
> A API utiliza **redes neurais YOLOv8** para identificar chamas em **imagens e vídeos**, auxiliando na **prevenção e resposta rápida** a incêndios florestais e urbanos.

---

## 🚀 **Visão Geral**

A **FireDetectAPI** oferece uma **API REST** que realiza **detecção automática de incêndios** em diferentes tipos de mídia.
Com base no modelo **YOLOv8n** (customizado como `fire_n`), o sistema é capaz de reconhecer regiões com presença de fogo, retornando **bounding boxes** e **confianças** em tempo real.

A aplicação pode ser integrada a sistemas de:

* Monitoramento ambiental e satelital;
* Drones e câmeras de vigilância;
* Plataformas de resposta emergencial.

---

## 🧠 **Arquitetura do Sistema**

```text
┌────────────────────────────┐
│          Cliente           │
│                            │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│         FastAPI            │
│  • Recebe requisições      │
│  • Processa uploads        │
│  • Retorna JSON + preview  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│        YOLOv8n Fire        │
│  • Modelo "fire_n.pt"      │
│  • Detecção de incêndio    │
│  • Saída: boxes + conf     │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│         OpenCV             │
│  • Pré-processamento       │
│  • Leitura de vídeo        │
│  • Renderização de frames  │
└────────────────────────────┘
```

---

## 🧩 **Tecnologias Utilizadas**

| Tecnologia                        | Função                             |
| --------------------------------- | ---------------------------------- |
| **Python 3.10+**                  | Linguagem principal                |
| **YOLOv8 (Ultralytics)**          | Detecção de incêndios              |
| **FastAPI**                       | Criação da API REST                |
| **OpenCV**                        | Manipulação de imagens e vídeos    |
| **Uvicorn**                       | Servidor ASGI de alto desempenho   |
| **Scikit-learn / pandas / NumPy** | Pré-processamento e modelagem      |
| **python-multipart**              | Upload de arquivos via API         |
| **Requests**                      | Testes de integração dos endpoints |

---

## ⚙️ **Endpoints Principais**

| Método | Rota        | Descrição                                         |
| ------ | ----------- | ------------------------------------------------- |
| `POST` | `/detect/`  | Envia uma imagem e retorna a detecção de incêndio |
| `POST` | `/predict/` | Prevê FWI                                         |
| `GET`  | `/health`   | Testa o status da API                             |

---

## 📦 **Modelo Utilizado**

* **Modelo base:** `fire_n.pt`
* **Arquitetura:** YOLOv8n
* **Tamanho:** ~3.2M parâmetros
* **Velocidade média:** ~90 FPS em GPU T4 Colab
* **Classes:** `fire`, `smoke`

---

## 🧪 **Demonstrações**

[https://github.com/user-attachments/assets/89f3b415-2904-4016-bc59-b8fec9157cba](https://github.com/user-attachments/assets/89f3b415-2904-4016-bc59-b8fec9157cba)

[https://github.com/user-attachments/assets/0af9ede1-c34c-4201-82e3-aef7b32efa31](https://github.com/user-attachments/assets/0af9ede1-c34c-4201-82e3-aef7b32efa31)

![image](https://github.com/user-attachments/assets/09cee042-9ce4-4889-8999-8bb0c3fa1c4e)

---

## 📈 **Resultados**

* **Acurácia média (mAP@50):** 92.4%
* **Tempo de inferência:** ~11 ms/frame
* **Taxa de falsos positivos:** < 5%

---

## 💡 **Aplicações Extras**

* Integração com **sensores IoT e satélites (FWI Index)**
* Envio de **alertas automáticos via WhatsApp e Telegram**
* Dashboard com mapas e **geolocalização de focos ativos**

---

## 🧭 **Guia Rápido — Execução e SDK**

### 🔹 Rodando a API

A API já pode ser iniciada diretamente com:

```bash
python app/main.py
```

A aplicação iniciará automaticamente o servidor FastAPI e carregará o modelo `fire_n.pt`.

### 🔹 Rodando no Google Colab

1. Clone o repositório:

```bash
!git clone REPO_URL
%cd FireDetectAPI
```

2. Instale as dependências:

```bash
!pip install -r requirements.txt
```

3. Rode a API:

```bash
!python app/main.py
```

### 🔹 Testando o SDK

O SDK do projeto foi feito para funcionar em **múltiplas plataformas**, como:

* **ESP32**  → edite e envie o arquivo `teste/client.cpp` para o microcontrolador.
* **Raspberry Pi / Jetson Nano / Desktop**  → execute:

```bash
python teste/test_api.py
```

Esses scripts se comunicam diretamente com a API para testar as detecções.

---

## 🌍 **Compatibilidade Ampla**

O projeto é democrático — o SDK pode rodar em diversas plataformas como **Raspberry Pi**, **Jetson Nano**, **ESP32**, **ambientes locais** e **nuvem**, permitindo integração com diferentes dispositivos e aplicações reais.
