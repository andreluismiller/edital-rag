import streamlit as st
from pathlib import Path
import sys

# Garante que a raiz do projeto esteja no path para importar o rag.py
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rag import RAGSystem, DEFAULT_QDRANT_PATH, DEFAULT_MODELS_DIR

st.set_page_config(
    page_title="Consulta ao Termo de Referência",
    page_icon="📄",
    layout="wide",
)

# --- Inicialização do sistema (cacheado para não recarregar a cada interação) ---
@st.cache_resource(show_spinner="Carregando modelos de embedding e conectando ao Qdrant...")
def load_rag_system():
    return RAGSystem(
        qdrant_path=str(DEFAULT_QDRANT_PATH),
        models_dir=str(DEFAULT_MODELS_DIR),
    )

try:
    rag = load_rag_system()
except Exception as e:
    st.error(f"Falha ao inicializar o sistema RAG: {e}")
    st.stop()

# --- Inicialização do estado da sessão ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Barra lateral: filtros e informações ---
with st.sidebar:
    st.header("Filtros de busca (opcional)")
    st.caption("Restrinja a recuperação para aumentar a precisão.")

    chunk_type = st.selectbox(
        "Tipo de trecho",
        options=[None, "clausula", "tabela", "lista", "secao"],
        format_func=lambda x: "Todos" if x is None else x.capitalize(),
    )
    item_number = st.text_input(
        "Item específico (ex.: 5.2.3.10)",
        placeholder="Deixe em branco para buscar em todo o documento",
    )
    norma = st.text_input(
        "Norma citada (ex.: NBR 5440)",
        placeholder="Filtra por normas no payload",
    )

    st.divider()
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**Sobre esta interface**\n\n"
        "As respostas são geradas exclusivamente a partir do Termo de Referência "
        "e citam o item correspondente. Os filtros acima restringem a busca híbrida "
        "(densa + esparsa) antes da geração."
    )

# --- Cabeçalho principal ---
st.title("📄 Consulta ao Termo de Referência")
st.caption(
    "Faça perguntas em linguagem natural sobre o edital de transformadores (Aracruz/ES). "
    "As respostas são baseadas no documento e citam o item correspondente."
)

# --- Renderiza o histórico de mensagens ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Se houver fontes salvas na mensagem, exibe em um expander
        if msg.get("sources"):
            with st.expander("Ver trechos consultados"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**Item {src.get('item_number', '?')}** "
                        f"({src.get('chunk_type', '?')}) — score `{src.get('score')}`  \n"
                        f"{src.get('section_path', '')}"
                    )
                    st.divider()

# --- Campo de entrada do chat ---
if prompt := st.chat_input("Ex.: Qual é o prazo máximo de entrega dos transformadores?"):
    # 1. Adiciona e exibe a pergunta do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Executa o pipeline RAG com os filtros da sidebar
    with st.chat_message("assistant"):
        with st.spinner("Consultando o Termo de Referência..."):
            try:
                result = rag.execute_rag_pipeline(
                    user_query=prompt,
                    chunk_type=chunk_type,
                    item_number=item_number.strip() or None,
                    norma=norma.strip() or None,
                )
                answer = result["answer"]
                sources = result.get("context_sources", [])

                st.markdown(answer)

                # Exibe as fontes usadas em um expander
                if sources:
                    with st.expander("Ver trechos consultados"):
                        for src in sources:
                            st.markdown(
                                f"**Item {src.get('item_number', '?')}** "
                                f"({src.get('chunk_type', '?')}) — score `{src.get('score')}`  \n"
                                f"{src.get('section_path', '')}"
                            )
                            st.divider()

                # Telemetria discreta no rodapé da mensagem
                st.caption(
                    f"⏱️ {result['response_time_seconds']}s · "
                    f"🎯 {result['total_tokens']} tokens · "
                    f"💵 ${result['cost_usd']:.6f}"
                )

                # 3. Salva a resposta e as fontes no histórico
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                st.error(f"Erro ao processar a pergunta: {e}")