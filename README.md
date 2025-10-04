# 🔥 API de Detecção de Incêndios com YOLO

Este projeto é o Trabalho de Conclusão de Curso (TCC) que consiste no desenvolvimento de uma API para detecção automática de incêndios em imagens e vídeos, utilizando a arquitetura de rede neural YOLO (You Only Look Once).

A aplicação tem como objetivo auxiliar no monitoramento e prevenção de incêndios florestais e urbanos, fornecendo uma ferramenta rápida e eficiente para a identificação de chamas em tempo real.

🚀 Tecnologias Utilizadas

- Python 3.10+

- YOLOv8 (Ultralytics) para detecção de incêndios

- FastAPI para construção da API REST

- OpenCV para manipulação de imagens e vídeos

- Uvicorn para servidor ASGI

- Scikit-learn, pandas e NumPy para pré-processamento e modelagem de dados

- Requests para testes de integração dos endpoints

- python-multipart para uploads de arquivos via API

https://github.com/user-attachments/assets/89f3b415-2904-4016-bc59-b8fec9157cba

## 📁 Estrutura do Projeto

```
DeepMeca-Fire/
├── app/
│   └── server.py           # API FastAPI com endpoints de predição e detecção
├── models/                 # Artefatos treinados (regressão, classificação e YOLO)
├── src/
│   └── models.py           # Classe utilitária para carregar os modelos
├── teste/
│   ├── local_model_tests.py  # Testes rápidos dos modelos locais
│   └── test_api.py           # Script de smoke test contra a API
├── Algerian_forest_fires_dataset_CLEANED.csv  # Dataset usado nos notebooks/modelos
├── Wildfire.ipynb / deepmeca.ipynb            # Notebooks exploratórios
└── README.md
```

## 📦 Pré-requisitos

- Python 3.10 ou superior
- pip 22+
- Sistema operacional com suporte a bibliotecas científicas (Linux, macOS ou Windows)
- Ambiente virtual recomendado (``venv`` ou ``conda``)

## 🛠️ Configuração do Ambiente

1. Clone o repositório e acesse a pasta do projeto:

   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd DeepMeca-Fire
   ```

2. (Opcional) Crie e ative um ambiente virtual:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. Instale as dependências necessárias:

   ```bash
   pip install fastapi "uvicorn[standard]" scikit-learn pandas numpy opencv-python ultralytics requests python-multipart
   ```

4. Garanta que a pasta ``models/`` contenha os arquivos de pesos:

   - ``classification.pkl`` – modelo de classificação (risco de incêndio)
   - ``regression.pkl`` – modelo de regressão (Fire Weather Index)
   - ``fire_n.pt`` – pesos do detector YOLO

## 🤖 Modelos e Dados

- Os arquivos ``classification.pkl`` e ``regression.pkl`` são carregados comprimidos em BZ2 e treinados a partir do dataset ``Algerian_forest_fires_dataset_CLEANED.csv``.
- O arquivo ``fire_n.pt`` é o modelo YOLOv8 treinado para detecção de focos de incêndio em imagens e vídeos.
- O módulo ``src/models.py`` disponibiliza a classe ``Model`` que centraliza o carregamento e cache dos artefatos para uso compartilhado por scripts e pela API.

## 🚀 Executando o Servidor

1. Certifique-se de que todas as dependências estejam instaladas e de que os modelos estejam na pasta ``models/``.
2. Inicie o servidor FastAPI com Uvicorn:

   ```bash
   uvicorn app.server:app --host 0.0.0.0 --port 8000
   ```

   ou execute diretamente o módulo do servidor:

   ```bash
   python -m app.server
   ```

3. A documentação interativa estará disponível em ``http://localhost:8000/docs`` (Swagger UI) e ``http://localhost:8000/redoc``.

## 🌐 Endpoints Disponíveis

| Método | Rota      | Descrição                                                                 |
| ------ | --------- | --------------------------------------------------------------------------- |
| GET    | `/health` | Verifica se a API está pronta para receber requisições.                     |
| POST   | `/predict`| Recebe atributos meteorológicos e retorna o FWI estimado e o risco (0 ou 1).|
| POST   | `/detect` | Recebe uma imagem (multipart/form-data) e retorna as detecções do YOLO.     |

### Payload de Exemplo – `/predict`

```json
{
  "Temperature": 31,
  "Ws": 14,
  "FFMC": 82.6,
  "DMC": 5.8,
  "ISI": 3.1
}
```

### Resposta de Exemplo – `/predict`

```json
{
  "fwi_estimation": 12.37,
  "fire_risk": 1,
  "status_label": "🔥 Perigo de incêndio"
}
```

## ✅ Testes Locais dos Modelos

O script ``teste/local_model_tests.py`` executa verificações rápidas para garantir que os modelos estão funcionando.

- Testar regressão e classificação (usa um exemplo padrão):

  ```bash
  python teste/local_model_tests.py
  ```

- Incluir um teste de imagem com o YOLO:

  ```bash
  python teste/local_model_tests.py --image caminho/para/imagem.jpg
  ```

- Incluir um teste de vídeo (analisando os 10 primeiros frames):

  ```bash
  python teste/local_model_tests.py --video caminho/para/video.mp4 --frames 10
  ```

## 🧪 Testes da API

Utilize ``teste/test_api.py`` para fazer um smoke test dos endpoints.

```bash
python teste/test_api.py --base-url http://127.0.0.1:8000 \
    --image-path caminho/para/imagem.jpg
```

- ``--payload`` aceita um JSON em texto para alterar os valores de entrada (ex.: ``--payload '{"Temperature": 28}'``).
- Se ``--image-path`` não for informado, apenas a rota ``/predict`` será exercitada.

## 🗂️ Dataset

- O dataset ``Algerian_forest_fires_dataset_CLEANED.csv`` contém as features meteorológicas utilizadas para treinar os modelos.
- O arquivo é carregado automaticamente pelos scripts para ajustar o ``StandardScaler`` utilizado na etapa de pré-processamento.

## 💡 Boas Práticas

- Mantenha os arquivos de modelo versionados e atualizados conforme novos treinamentos.
- Atualize o README com novas instruções sempre que os scripts ou endpoints forem modificados.
- Utilize ambientes virtuais isolados para evitar conflitos de dependências.

## 📄 Licença e Créditos

Este projeto é desenvolvido como parte do TCC e utiliza componentes open-source.
Consulte as licenças individuais das bibliotecas utilizadas para mais detalhes.
