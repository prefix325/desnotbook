# Machine Learning Project Template

Este é um template estruturado para projetos de Machine Learning em Python, seguindo as melhores práticas da indústria.

## Instalação

1. Clone o repositório
2. Crie um ambiente virtual: `python -m venv venv`
3. Ative o ambiente: `source venv/bin/activate` (Linux/Mac) ou `venv\Scripts\activate` (Windows)
4. Instale as dependências: `pip install -r requirements.txt`
5. Instale o projeto: `pip install -e .`

## Uso

1. Coloque seus dados brutos em `data/raw/`
2. Execute os scripts de processamento em `src/data/`
3. Desenvolva features em `src/features/`
4. Treine modelos usando `src/models/`
5. Visualize resultados com `src/visualization/`

## Comandos Úteis

- `make data`: Processa dados brutos
- `make features`: Gera features
- `make train`: Treina modelo
- `make predict`: Faz predições
- `make test`: Executa testes
- `make clean`: Limpa arquivos temporários

## Contribuição

1. Faça um fork do projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request
