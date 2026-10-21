# RAG — Termo de Referência do Edital de Transformadores (Aracruz/ES)

Sistema de Recuperação Aumentada por Geração (RAG) que permite a colaboradores
de uma indústria de transformadores consultarem, em linguagem natural, o Termo
de Referência do edital de compra de transformadores elétricos da Prefeitura
Municipal de Aracruz/ES — sem precisar abrir e ler o documento inteiro.

O sistema é **local-first**, roda em Codespaces com recursos limitados e usa
**Qdrant local em disco** + **FastEmbed (ONNX)** + **Groq (LLM)**.

---

## Sumário

1. [Objetivo e público-alvo](#1-objetivo-e-público-alvo)
2. [Visão geral da arquitetura](#2-visão-geral-da-arquitetura)
3. [Pré-requisitos](#3-pré-requisitos)
4. [Instalação](#4-instalação)
5. [Estrutura do projeto](#5-estrutura-do-projeto)
6. [Pipeline em detalhes](#6-pipeline-em-detalhes)
   - [Etapa 1 — Pré-processamento do Markdown](#etapa-1--pré-processamento-do-markdown)
   - [Etapa 2 — Parse hierárquico e chunking](#etapa-2--parse-hierárquico-e-chunking)
   - [Etapa 3a — Download dos modelos](#etapa-3a--download-dos-modelos)
   - [Etapa 3b — Embedding e ingestão no Qdrant](#etapa-3b--embedding-e-ingestão-no-qdrant)
   - [Etapa 4 — Busca, RAG e resposta](#etapa-4--busca-rag-e-resposta)
7. [Schema do chunk](#7-schema-do-chunk)
8. [Modelo de dados no Qdrant](#8-modelo-de-dados-no-qdrant)
9. [Módulo `rag.py` em detalhes](#9-módulo-ragpy-em-detalhes)
10. [Estratégias de eficiência para Codespaces](#10-estratégias-de-eficiência-para-codespaces)
11. [Exemplos de perguntas suportadas](#11-exemplos-de-perguntas-suportadas)
12. [Solução de problemas](#12-solução-de-problemas)
13. [Limitações e roadmap](#13-limitações-e-roadmap)
14. [Referências](#14-referências)

---

## 1. Objetivo e público-alvo

### Objetivo

Disponibilizar uma interface de perguntas e respostas, em português, sobre o
**Termo de Referência** de um edital público de aquisição de transformadores
elétricos. O sistema recupera trechos relevantes do documento, monta um contexto
controlado e delega a resposta a um LLM, sempre citando o item do Termo em que
a informação se baseia.

### Público-alvo

| Setor | Uso típico |
|---|---|
| **Engenharia** | Consultar especificações técnicas (normas, óleo isolante, buchas, ensaios, taps). |
| **Comercial** | Verificar quantidades, valores unitários e totais, prazos, condições de garantia e pagamento. |
| **PMO** | Acompanhar obrigações da contratada, sanções, recebimento provisório/definitivo, gestão e fiscalização. |

### Escopo

- **Entrada**: Termo de Referência do edital (Markdown convertido de PDF).
- **Saída**: respostas em Markdown, com citação de item e telemetria (tokens, custo, tempo).
- **Fora de escopo**: consulta a outros anexos do edital; extração automática de PDF; UI web (o projeto entrega CLI).

---

## 2. Visão geral da arquitetura

```
┌─────────────────────┐
│ PDF do Termo        │
└─────────┬───────────┘
          │  (conversão manual ou externa)
          ▼
┌─────────────────────────────┐
│ edital_aracruz_es.md        │  data/processed/
└─────────┬───────────────────┘
          │  Etapa 1 — preprocess_md.py
          ▼
┌─────────────────────────────┐
│ edital_aracruz_es_clean.md  │  data/processed/
└─────────┬───────────────────┘
          │  Etapa 2 — chunk_md.py
          ▼
┌─────────────────────────────┐
│ edital_aracruz_es_chunks    │  data/processed/
│ .jsonl                      │
└─────────┬───────────────────┘
          │  Etapa 3a — download_models.py
          │  Etapa 3b — embed_qdrant.py
          ▼
┌─────────────────────────────┐      ┌───────────────────────┐
│ Qdrant local (disco)        │◄─────│ models/ (ONNX em      │
│ data/qdrant_db/             │      │ cache local)          │
└─────────┬───────────────────┘      └───────────────────────┘
          │  Etapa 4 — rag.py
          ▼
┌─────────────────────────────┐
│ Resposta (Markdown) +       │
│ fontes + telemetria         │
└─────────────────────────────┘
```

**Modelos e serviços**

| Componente | Escolha | Papel |
|---|---|---|
| Embedding denso | `intfloat/multilingual-e5-small` (384d) | Similaridade semântica multilíngue (pt) |
| Embedding esparso | `Qdrant/bm25` | Busca lexical (siglas, valores, normas) |
| Banco vetorial | Qdrant local (`path`) | Persistência sem servidor |
| LLM | Groq `llama-3.3-70b-versatile` | Geração da resposta final |
| Gerenciador de pacotes | `uv` | Ambiente reprodutível |

**Estratégia de recuperação**: busca híbrida (densa + esparsa) com fusão
**Reciprocal Rank Fusion (RRF)**. Filtros por `chunk_type`, `item_number`,
`doc_id`, `entities` e `normas_citadas` podem ser aplicados antes ou durante
a busca.

---

## 3. Pré-requisitos

- Python **3.11+**
- [`uv`](https://docs.astral.sh/uv/) instalado
- Chave de API do **Groq** (gratuita em <https://console.groq.com>)
- **~2 GB de disco** livres (modelos ONNX + Qdrant + caches)
- Conexão de rede apenas nas etapas 3a (download) e 4 (LLM)

---

## 4. Instalação

### 4.1 Clonar e instalar dependências

```bash
git clone <url-do-repo>
cd edital-rag
uv sync
```

### 4.2 Configurar a chave do Groq

Crie um arquivo `.env` na raiz:

```bash
echo 'GROQ_API_KEY="sua-chave-aqui"' > .env
```

> O `rag.py` carrega automaticamente o `.env` via `python-dotenv`. Alternativa
> robusta: `uv run --env-file .env python scripts/rag.py "..."`.

### 4.3 (Opcional) Modo offline

Para garantir que o FastEmbed não tente contatar a Hugging Face durante a etapa 4:

```bash
echo 'HF_HUB_OFFLINE=1' >> .env
```

---

## 5. Estrutura do projeto

```
edital-rag/
├── data/
│   ├── processed/
│   │   ├── edital_aracruz_es.md               # entrada (Markdown bruto)
│   │   ├── edital_aracruz_es_clean.md         # saída da etapa 1
│   │   └── edital_aracruz_es_chunks.jsonl     # saída da etapa 2
│   └── qdrant_db/                             # Qdrant local (etapa 3b)
├── models/                                    # cache ONNX (etapa 3a)
├── scripts/
│   ├── preprocess_md.py                       # etapa 1
│   ├── chunk_md.py                            # etapa 2
│   ├── download_models.py                     # etapa 3a
│   ├── embed_qdrant.py                        # etapa 3b
│   └── rag.py                                 # etapa 4
├── .env
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 6. Pipeline em detalhes

Cada etapa é um script independente, com entrada e saída em arquivo. Isso
permite reexecutar apenas o trecho necessário (por exemplo, reingerir após
trocar de modelo) sem refazer tudo.

### Etapa 1 — Pré-processamento do Markdown

**Script**: `scripts/preprocess_md.py`

**Entrada**: `data/processed/edital_aracruz_es.md`
**Saída**: `data/processed/edital_aracruz_es_clean.md`

**Objetivo**: transformar o Markdown bruto (gerado a partir do PDF) em um
documento limpo, com tabelas íntegras e numeração normalizada, pronto para o
parser hierárquico da etapa seguinte.

**O que faz**:

| Operação | Descrição |
|---|---|
| Remove `<!-- image -->` | Placeholders de imagem do PDF são descartados; a contagem é registrada no log. |
| Funde tabelas fragmentadas | Fragmentos separados por linhas em branco, comentários HTML, headings de documento (`ANEXO I`, `TERMO DE REFERÊNCIA`) ou placeholders já removidos são unidos quando o cabeçalho é compatível ou quando a linha parece continuação. |
| Corrige quebras dentro de células | Uma linha de tabela cuja primeira célula está vazia (e maioria das demais) é anexada à linha anterior, célula a célula. |
| Reposiciona headings "ruidosos" | Headings que estavam no meio de tabelas (como `## ANEXO I - TERMO DE REFERÊNCIA`) são movidos para o topo. |
| Normaliza títulos | `## 5 - DEFINIÇÕES` → `## 5. DEFINIÇÕES`; espaços finais são removidos. |
| Preserva valores e normas | Um verificador conta valores monetários, potências (kVA), tensões (kV), prazos e normas antes/depois e reporta perdas. |
| Reporta anomalias | Linhas com número de colunas inconsistente, células iniciadas por `NBR` quando a anterior termina em `conforme`, seções duplicadas. |

**Execução**:

```bash
uv run python scripts/preprocess_md.py
```

**Limitações conhecidas**:

- O caso específico "célula do item 2 começa com `NBR`" (provável `NBR` que
  pertencia ao item 1) é apenas reportado, não corrigido — exige revisão humana.
- As listas de ensaios são mantidas como listas (não convertidas em tabela).

---

### Etapa 2 — Parse hierárquico e chunking

**Script**: `scripts/chunk_md.py`

**Entrada**: `data/processed/edital_aracruz_es_clean.md`
**Saída**: `data/processed/edital_aracruz_es_chunks.jsonl`

**Objetivo**: transformar o Markdown limpo em uma lista de chunks JSONL,
respeitando a hierarquia numérica do documento (`5 > 5.2 > 5.2.3 > 5.2.3.10`)
e produzindo um schema estável para ingestão.

**Estratégia de chunking**:

A estratégia **não é chunk fixo por tamanho**. Ela é hierárquica, orientada à
estrutura, com parent-child, tabelas atômicas e listas com cabeçalho repetido.

| Tipo de chunk | Regra |
|---|---|
| **`secao` (pai)** | Seção com filhos. Guarda o texto próprio (prosa antes dos filhos) ou um sumário dos filhos. Serve de contexto ampliado (small-to-big). |
| **`clausula` (filho)** | Cláusula numerada ou parágrafo isolado dentro de uma seção folha. |
| **`tabela` (filho)** | Tabela atômica com cabeçalho. Se > 20 linhas de dados, é dividida em blocos com cabeçalho repetido e `row_range` específico. |
| **`lista` (filho)** | Lista atômica se ≤ 8 itens. Acima disso, blocos de 8 com cabeçalho `[N. Título]` no topo de cada sub-bloco. |

**Decisões de design**:

- **IDs determinísticos**: `tr-{item}-pai` (seção), `tr-{item}-{NNN}` (cláusula), `tr-{item}-tbl-{NN}` (tabela), `tr-{item}-lst-{NN}` (lista).
- **`section_path` compacto por padrão**: `ANEXO I > Termo de Referência > 5. DEFINIÇÕES > 5.2.3.10 Líquido Isolante` (2 níveis + prefixo). Use `--full-path` para todos os ancestrais.
- **`parent_id`**: sempre o pai numérico direto (`tr-5.2.3-pai` para `5.2.3.10`).
- **Enriquecimento automático** de cada chunk:
  - `entities`: léxico de domínio (óleo isolante, comutador, bucha, tanque etc.) + acrônimos (exceto prefixos de normas).
  - `normas_citadas`: NBR/IEC/ISO/EDP/ANP.
  - `valores`: R$, kVA, kV, %, prazos.
  - `referencias_resolvidas`: referências internas (`conforme item N`) que apontam para seções existentes. Refs precedidas por norma (ex.: `NBR 5440, item 6.1`) são descartadas como externas.

**Execução**:

```bash
uv run python scripts/chunk_md.py
# ou, para caminho de seção completo:
uv run python scripts/chunk_md.py --full-path --max-list-items 10
```

**Exemplo de chunk gerado** (para a seção `5.2.3.10 Líquido Isolante`):

```json
{
  "chunk_id": "tr-5.2.3.10-001",
  "doc_id": "edital_aracruz_es",
  "doc_title": "Termo de Referência — Aracruz/ES",
  "section_path": "ANEXO I - TERMO DE REFERÊNCIA > 5. DEFINIÇÕES > 5.2.3.10. Líquido Isolante",
  "item_number": "5.2.3.10",
  "chunk_type": "clausula",
  "parent_id": "tr-5.2.3-pai",
  "child_ids": [],
  "table_id": null,
  "row_range": null,
  "entities": ["ANP", "PCB", "óleo isolante"],
  "normas_citadas": ["NBR 8371", "ES.DT.PDN.01.05.015"],
  "valores": ["0,3 (± 0,03) %"],
  "referencias_resolvidas": [],
  "source_anchor": "5.2.3.10",
  "texto": "5.2.3.10. Líquido Isolante\n\nO óleo dos transformadores..."
}
```

---

### Etapa 3a — Download dos modelos

**Script**: `scripts/download_models.py`

**Entrada**: rede (Hugging Face)
**Saída**: `models/` (cache ONNX)

**Objetivo**: baixar uma única vez os modelos do FastEmbed para `models/`,
deixando a etapa 3b e a etapa 4 totalmente offline e determinísticas.

**Modelos padrão**:

| Tipo | Modelo | Dimensão |
|---|---|---|
| Denso | `intfloat/multilingual-e5-small` | 384 |
| Esparso | `Qdrant/bm25` | vocabulário BM25 |

**Execução**:

```bash
uv run python scripts/download_models.py
# para usar o modelo "large" (1024d, mais qualidade e mais RAM):
uv run python scripts/download_models.py --dense-model intfloat/multilingual-e5-large
```

**Por que modelos pequenos por padrão**: o `e5-small` tem 384 dimensões contra
1024 do `large`. Vetores menores reduzem memória, tempo de ingestão e uso de
disco — requisitos críticos em Codespaces.

---

### Etapa 3b — Embedding e ingestão no Qdrant

**Script**: `scripts/embed_qdrant.py`

**Entrada**: `data/processed/edital_aracruz_es_chunks.jsonl`, `models/`
**Saída**: `data/qdrant_db/` (Qdrant local em disco)

**Objetivo**: gerar embeddings densos e esparsos para cada chunk e inserir no
Qdrant com payload indexado, habilitando busca híbrida filtrada.

**O que faz**:

1. Carrega os modelos **exclusivamente** de `models/` (sem rede).
2. Cria a coleção `edital_aracruz_es` com dois vetores nomeados:
   - `dense`: 384d, distância **Cosine**.
   - `sparse`: BM25 com `Modifier.IDF` (necessário para cálculo correto do BM25 no query time).
3. Cria índices de payload para filtros:

   | Campo | Schema |
   |---|---|
   | `doc_id` | KEYWORD |
   | `item_number` | KEYWORD |
   | `chunk_type` | KEYWORD |
   | `section_path` | TEXT |
   | `parent_id` | KEYWORD |
   | `source_anchor` | KEYWORD |
   | `entities` | KEYWORD |
   | `normas_citadas` | KEYWORD |

4. Processa o JSONL **em lotes** (padrão: 16 chunks), mantendo consumo de RAM constante.
5. Faz **upsert idempotente**: o `id` do ponto é `UUID5(chunk_id)`, então reprocessar sobrescreve em vez de duplicar.

**Texto vetorizado**: `section_path` + `texto` (o caminho da seção entra no
embedding para melhorar a recuperação).

**Execução**:

```bash
# ingestão padrão (idempotente)
uv run python scripts/embed_qdrant.py

# recriar a coleção do zero
uv run python scripts/embed_qdrant.py --recreate

# ajustar batch para CodeSpaces com pouca memória
uv run python scripts/embed_qdrant.py --batch-size 8
```

**Idempotência**:

| Cenário | Comportamento |
|---|---|
| Rodar duas vezes sem `--recreate` | Sobrescreve pontos com o mesmo `chunk_id`. |
| Rodar com `--recreate` | Apaga a coleção e reinsere. Use ao trocar de modelo (dimensão muda). |
| Rodar sem rede, com `models/` já populado | Funciona normalmente. |

---

### Etapa 4 — Busca, RAG e resposta

**Script**: `scripts/rag.py`

**Entrada**: pergunta em linguagem natural
**Saída**: resposta em Markdown + fontes + telemetria

**Objetivo**: receber uma pergunta, recuperar os chunks mais relevantes
(busca híbrida com filtros opcionais), montar o prompt com contexto
controlado e gerar a resposta via Groq.

**Componentes**:

| Componente | O que faz |
|---|---|
| `hybrid_search()` | Busca densa + esparsa em paralelo (`Prefetch`), fusão via **RRF**. |
| `_build_filter()` | Constrói `Filter` a partir de `chunk_type`, `doc_id`, `item_number`, `entity`, `norma`. |
| `build_context()` | Formata os resultados como blocos numerados com item, tipo, score, normas e valores. |
| `build_prompt()` | Instruções fixas + contexto + pergunta do usuário. |
| `generate_response()` | Chama o Groq e retorna resposta + tokens. |
| `calculate_cost()` | Calcula custo em USD por 1M de tokens. |
| `execute_rag_pipeline()` | Orquestra tudo e devolve dict com telemetria. |

**Persona do LLM**: especialista em editais de compras públicas de
transformadores, atendendo Engenharia, Comercial e PMO. Regras incluem:
responder em pt-BR, citar o item do Termo, preservar valores exatos e nunca
inventar informação fora do contexto.

**Execução**:

```bash
# pergunta livre
uv run python scripts/rag.py "Qual é o prazo máximo de entrega dos transformadores?"

# restringir a tabelas (útil para "quais fabricantes homologados para 300 kVA?")
uv run python scripts/rag.py "..." --chunk-type tabela

# filtrar por norma
uv run python scripts/rag.py "..." --norma "NBR 5440"

# filtrar por item exato
uv run python scripts/rag.py "..." --item-number 5.2.3.10

# filtrar por entidade
uv run python scripts/rag.py "..." --entity "ensaios de rotina"
```

---

## 7. Schema do chunk

Cada linha do `*_chunks.jsonl` segue exatamente o schema abaixo. Ele foi
projetado para servir tanto à ingestão quanto à busca filtrada e à auditoria.

| Campo | Tipo | Descrição |
|---|---|---|
| `chunk_id` | string | ID determinístico (`tr-{item}-{NNN}`, `tr-{item}-pai`, `tr-{item}-tbl-{NN}`, `tr-{item}-lst-{NN}`). |
| `doc_id` | string | Identificador do documento (`edital_aracruz_es`). |
| `doc_title` | string | Título legível. |
| `section_path` | string | Caminho da seção (`ANEXO I > ... > 5.2.3.10 Título`). |
| `item_number` | string | Número do item (`5.2.3.10`). |
| `chunk_type` | enum | `secao`, `clausula`, `tabela`, `lista`. |
| `parent_id` | string \| null | Chunk pai (`tr-5.2.3-pai`), para small-to-big. |
| `child_ids` | string[] | Filhos diretos (populado apenas em `secao`). |
| `table_id` | string \| null | ID lógico da tabela (todos os sub-blocos de uma mesma tabela compartilham). |
| `row_range` | [int, int] \| null | Faixa de linhas de dados coberta pelo chunk. |
| `entities` | string[] | Termos de domínio + acrônimos. |
| `normas_citadas` | string[] | NBR/IEC/ISO/EDP/ANP. |
| `valores` | string[] | R$, kVA, kV, %, prazos. |
| `referencias_resolvidas` | dict[] | `{texto, item, titulo}` para refs internas. |
| `source_anchor` | string | Número do item, para âncoras na UI. |
| `texto` | string | Conteúdo bruto do chunk (é o que vai ao LLM). |

---

## 8. Modelo de dados no Qdrant

### Coleção

```python
collection_name = "edital_aracruz_es"

vectors_config = {
    "dense": VectorParams(size=384, distance=Distance.COSINE)
}

sparse_vectors_config = {
    "sparse": SparseVectorParams(modifier=Modifier.IDF)
}
```

### Ponto (ponto típico no Qdrant)

```json
{
  "id": "uuid5(chunk_id)",
  "vector": {
    "dense": [0.012, -0.034, ...],
    "sparse": {"indices": [12, 45, 1023], "values": [0.8, 0.6, 0.4]}
  },
  "payload": {
    "chunk_id": "tr-5.2.3.10-001",
    "doc_id": "edital_aracruz_es",
    "item_number": "5.2.3.10",
    "chunk_type": "clausula",
    "section_path": "...",
    "parent_id": "tr-5.2.3-pai",
    "entities": ["óleo isolante", "PCB"],
    "normas_citadas": ["NBR 8371"],
    "valores": ["0,3 (± 0,03) %"],
    "texto": "..."
  }
}
```

### Por que dois vetores?

| Vetor | Vantagem | Caso de uso |
|---|---|---|
| **Denso** | Captura paráfrases e semântica (“prazo de entrega” ≈ “quando deve ser entregue”). | Perguntas em linguagem natural. |
| **Esparso (BM25)** | Casa termos exatos (siglas, valores, normas, nomes de fabricante). | “NBR 5440”, “300 kVA”, “R$ 37.882,57”. |

A fusão por **RRF** combina os dois rankings sem precisar calibrar escalas.

---

## 9. Módulo `rag.py` em detalhes

### Fluxo interno de `execute_rag_pipeline()`

```
1. check_collection_exists()       → falha rápido se a coleção não existe
2. hybrid_search(query, filters)   → top-k via RRF com filtros
3. build_prompt(query, results)    → instruções + contexto + pergunta
4. generate_response(prompt)       → Groq; retorna resposta + tokens
5. calculate_cost(tokens)          → custo em USD
6. return dict com telemetria      → resposta, fontes, tokens, custo, tempo
```

### Filtros disponíveis (parâmetros de `hybrid_search` e `execute_rag_pipeline`)

| Parâmetro | Tipo | Efeito |
|---|---|---|
| `chunk_type` | `clausula` \| `tabela` \| `lista` \| `secao` | Restringe a busca a um tipo. |
| `doc_id` | string | Restringe a um documento (útil quando houver múltiplos editais). |
| `item_number` | string | Restringe a um item específico (`5.2.3.10`). |
| `entity` | string | Restringe a chunks que contenham a entidade (`óleo isolante`). |
| `norma` | string | Restringe a chunks que citem a norma (`NBR 5440`). |

Filtros são aplicados **tanto no `Prefetch` quanto na `FusionQuery`**, o que
garante consistência entre as duas pernas da busca.

### Utilidades

- `list_distinct(field)`: lista valores distintos de um campo keyword, útil
  para popular seletores numa futura UI (`chunk_type`, `entities`,
  `normas_citadas`).
- `build_context()`: já formata cada fonte com item, tipo, score, normas e
  valores — o LLM recebe contexto autoexplicativo.
- `calculate_cost()`: usa a tabela de preços do Groq; fácil de estender para
  outros modelos.

### Telemetria retornada

```python
{
  "question": "...",
  "answer": "...",                     # Markdown, com citação do item
  "filters": {...},                    # filtros aplicados
  "model_used": "llama-3.3-70b-versatile",
  "instructions": "...",               # persona usada
  "prompt": "...",                     # prompt completo (auditoria)
  "response_time_seconds": 2.13,
  "prompt_tokens": 1450,
  "completion_tokens": 220,
  "total_tokens": 1670,
  "cost_usd": 0.00093,
  "context_sources": [                 # o que foi efetivamente usado
    {"score": 0.031, "item_number": "1.2", "chunk_type": "clausula",
     "section_path": "...", "normas_citadas": []}
  ]
}
```

---

## 10. Estratégias de eficiência para Codespaces

O projeto foi desenhado para rodar em containers modestos (2 vCPU, 4–8 GB
de RAM). As principais escolhas:

| Estratégia | Efeito |
|---|---|
| **Modelo denso `e5-small` (384d)** | Vetores 2,6× menores que `e5-large` (1024d). Menos RAM, disco e tempo. |
| **`local_files_only` removido; `HF_HUB_OFFLINE=1` opcional** | Evita checagens de rede desnecessárias. |
| **Streaming do JSONL em lotes** (`iter_chunks`) | Consumo de RAM constante, independente do tamanho do arquivo. |
| **Batch size conservador (16)** | Ajustável para 8 ou 4 em containers muito pequenos. |
| **Qdrant local em disco** (`QdrantClient(path=...)`) | Persistência sem servidor, libera RAM entre execuções. |
| **Upsert idempotente (UUID5)** | Não duplica pontos; pode reexecutar à vontade. |
| **Separação download / ingestão** | Download é feito uma única vez; ingestão pode reexecutar offline. |
| **`on_disk_payload` não forçado** | Não é suportado de forma confiável no modo local do Qdrant Client; evitá-lo evita erros silenciosos. |

**Se ainda faltar memória**:

1. Reduza `--batch-size` para 8 ou 4 na ingestão.
2. Reduza `--top-k` para 3 no `rag.py`.
3. Use `--chunk-type` para restringir o escopo antes da busca.
4. Considere `intfloat/multilingual-e5-small` com `HF_HUB_OFFLINE=1` e
   `--dense-model` explícito.

---

## 11. Exemplos de perguntas suportadas

### Engenharia

```bash
uv run python scripts/rag.py "Quais normas técnicas devem ser atendidas na fabricação?"
uv run python scripts/rag.py "Qual óleo isolante é permitido e quais requisitos ele deve atender?"
uv run python scripts/rag.py "Quais ensaios de rotina devem ser executados?" --chunk-type lista
uv run python scripts/rag.py "Quais os taps do enrolamento primário?" --item-number 5.2.3.4
uv run python scripts/rag.py "Requisitos de buchas de alta tensão" --norma "NBR 5440"
```

### Comercial

```bash
uv run python scripts/rag.py "Qual é o valor total estimado da aquisição?"
uv run python scripts/rag.py "Quais as quantidades e valores unitários por item?" --chunk-type tabela
uv run python scripts/rag.py "Qual é o prazo de garantia dos transformadores?"
uv run python scripts/rag.py "Quais fabricantes estão homologados para 300 kVA?" --chunk-type tabela
uv run python scripts/rag.py "Como será feito o pagamento?" --item-number 10
```

### PMO

```bash
uv run python scripts/rag.py "Como funciona o recebimento provisório e definitivo?"
uv run python scripts/rag.py "Quais sanções administrativas podem ser aplicadas?" --chunk-type lista
uv run python scripts/rag.py "Quem fiscaliza o contrato e quais são as atribuições?"
uv run python scripts/rag.py "Qual é o prazo de vigência do contrato e como será o reajuste?"
uv run python scripts/rag.py "O que acontece se a empresa não entregar no prazo?" --entity garantia
```

---

## 12. Solução de problemas

### `RuntimeError: Modelos de embedding não encontrados no cache local`

**Causa**: `models/` está vazio ou o `cache_dir` não foi respeitado.

**Solução**:

```bash
uv run python scripts/download_models.py
# ou, se o cache foi para ~/.cache/huggingface:
HF_HOME="$PWD/models" uv run python scripts/download_models.py
```

Confirme com:

```bash
find models -name "*.onnx"
```

### `A coleção 'edital_aracruz_es' não foi encontrada`

**Causa**: o Qdrant ainda não foi populado.

**Solução**:

```bash
uv run python scripts/chunk_md.py
uv run python scripts/embed_qdrant.py
```

### Resposta vaga ou "não encontrado"

**Causas possíveis**:

1. `--top-k` baixo → aumente para 8–10.
2. O chunk relevante não foi recuperado → use filtros (`--item-number`,
   `--chunk-type`) para restringir.
3. O modelo denso não é o mesmo da ingestão → verifique
   `--dense-model` no `rag.py` e reingira se mudou.

### Erro `LocalEntryNotFoundError` no `rag.py`

**Causa**: `local_files_only=True` (já removido na versão atual) ou rede
indisponível sem `models/` populado.

**Solução**: rode `download_models.py` com `HF_HOME` apontando para `models/`.

### `GROQ_API_KEY` não encontrada

**Solução**:

```bash
echo 'GROQ_API_KEY="sua-chave"' > .env
# ou
uv run --env-file .env python scripts/rag.py "..."
```

---

## 13. Limitações e roadmap

### Limitações atuais

- **Um documento por vez**: a coleção é por edital. Suporte a múltiplos editais
  exigiria `doc_id` no filtro (já previsto) e uma etapa de roteamento.
- **Sem UI web**: o projeto entrega CLI. `list_distinct()` já prepara os campos
  para popular seletores.
- **Sem small-to-big automático**: o `parent_id` está no payload, mas
  `rag.py` não expande para o pai automaticamente. Pode ser adicionado em
  `build_context()`.
- **Sem feedback loop**: a resposta não é persistida para avaliação offline.
  `context_sources` já traz tudo que seria necessário para auditoria.
- **`e5-small` com prefixo não aplicado**: o FastEmbed não prefixa `query:` /
  `passage:` por padrão. Se a qualidade cair, habilitar
  `query_embed()`/`passage_embed()` **em ambos** ingestão e busca e reingerir.

### Roadmap sugerido

1. **Small-to-big no `rag.py`**: expandir cada `clausula`/`tabela` para seu
   `parent_id` antes de montar o contexto.
2. **Avaliação automatizada**: usar as 40 perguntas do briefing como gold set;
   medir recall@k, precisão do contexto e fidelidade da resposta.
3. **UI web (Streamlit/Gradio)**: usar `list_distinct()` para popular filtros.
4. **Suporte a múltiplos documentos**: roteamento por `doc_id`.
5. **Persistência de feedback**: gravar `question`, `answer`,
   `context_sources`, `model_used` para auditoria.
6. **Fallback de modelo**: se `e5-small` não recuperar bem termos exatos,
   considerar `intfloat/multilingual-e5-large` em produção, mantendo
   `e5-small` em desenvolvimento.

---

## 14. Referências

- **Qdrant — Hybrid Search**: <https://qdrant.tech/documentation/concepts/hybrid-queries/>
- **Qdrant — Payload Indexing**: <https://qdrant.tech/documentation/concepts/indexing/>
- **FastEmbed — Supported Models**: <https://qdrant.github.io/fastembed/examples/Supported_Models/>
- **FastEmbed — Repo**: <https://github.com/qdrant/fastembed>
- **Groq — API OpenAI-compatible**: <https://console.groq.com/docs/openai>
- **uv — Docs**: <https://docs.astral.sh/uv/>
- **Lei Federal nº 14.133/2021** (Nova Lei de Licitações) — citada no Termo.
- **Documento fonte**: `data/processed/edital_aracruz_es.md` (Termo de
  Referência — Aracruz/ES, 02/03/2026).

---

## Licença

Uso interno. Consulte o responsável pelo projeto antes de redistribuir.