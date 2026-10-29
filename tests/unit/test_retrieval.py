import pathlib
import pytest

from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG

try:
    from multi_doc_chat.exception.custom_exception import DocumentPortalException as DPE1
except ImportError:
    DPE1 = Exception

try:
    from exception.custom_exception import DocumentPortalException as DPE2
except ImportError:
    DPE2 = Exception

ExpectedException = (DPE1, DPE2) if DPE1 is not DPE2 else DPE1


def test_conversationalrag_error_handling(tmp_dirs, stub_model_loader):
    rag = ConversationalRAG(session_id="s1")
    with pytest.raises(ExpectedException) as exc_info:
        rag.invoke("hello")
    assert "DocumentPortalException" in type(exc_info.value).__name__

    with pytest.raises(ExpectedException) as exc_info:
        rag.load_retriever_from_faiss(index_path="faiss_index/does_not_exist")
    assert "DocumentPortalException" in type(exc_info.value).__name__