import os
import time

import streamlit as st
from werkzeug.security import check_password_hash, generate_password_hash

from db import create_user, get_history, get_user, init_db, save_history
from predict import get_prediction_details, predict_batch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "vectorizer.pkl")

init_db()
st.set_page_config(page_title="Büyüteç Haber Ajansı", page_icon="🔎", layout="wide")


def auth_screen() -> None:
    st.title("🔐 Giriş")
    login_tab, register_tab = st.tabs(["Giriş Yap", "Kayıt Ol"])

    with login_tab:
        username = st.text_input("Kullanıcı adı", key="login_username")
        password = st.text_input("Şifre", type="password", key="login_password")
        if st.button("Giriş Yap", use_container_width=True):
            user = get_user(username.strip())
            if user and check_password_hash(user["password_hash"], password):
                st.session_state["user"] = username.strip()
                st.success("Giriş başarılı")
                st.rerun()
            st.error("Kullanıcı adı veya şifre hatalı")

    with register_tab:
        new_user = st.text_input("Yeni kullanıcı adı", key="reg_username")
        new_pass = st.text_input("Yeni şifre", type="password", key="reg_password")
        if st.button("Kayıt Oluştur", use_container_width=True):
            if len(new_user.strip()) < 3 or len(new_pass) < 6:
                st.warning("Kullanıcı adı en az 3, şifre en az 6 karakter olmalı.")
            else:
                ok = create_user(new_user.strip(), generate_password_hash(new_pass))
                if ok:
                    st.success("Kayıt oluşturuldu. Giriş yapabilirsiniz.")
                else:
                    st.error("Bu kullanıcı adı zaten kayıtlı.")

    if st.button("Misafir olarak devam et", use_container_width=True):
        st.session_state["user"] = "guest"
        st.rerun()




def render_sidebar() -> None:
    st.sidebar.header("Teknolojiler")
    st.sidebar.markdown("- Streamlit")
    st.sidebar.markdown("- Scikit-learn")
    st.sidebar.markdown("- SQLite")

def main() -> None:
    render_sidebar()

    if "user" not in st.session_state:
        auth_screen()
        return

    user = st.session_state["user"]
    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown(f"**Logged in as: {user}**")
    with top_right:
        if st.button("Çıkış"):
            st.session_state.clear()
            st.rerun()

    st.title("Büyüteç Haber Ajansı")
    model_ready = os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
    if not model_ready:
        st.error("Model dosyaları eksik.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["✍️ Metin Analizi", "📊 Toplu Analiz", "📋 Geçmiş"])

    with tab1:
        text = st.text_area("Haber metni", height=250)
        if st.button("ANALİZ ET", disabled=not text.strip(), use_container_width=True):
            with st.spinner("Analiz ediliyor..."):
                time.sleep(0.2)
                details = get_prediction_details(text)
            st.success(f"Sonuç: {details['label']}")
            st.write(f"Güven: %{details['confidence']:.2f}")
            st.progress(details["confidence"] / 100)
            if user != "guest":
                save_history(
                    username=user,
                    text_input=text,
                    prediction_label="FAKE" if details["is_fake"] else "REAL",
                    confidence_score=float(details["confidence"]),
                )

    with tab2:
        bulk_text = st.text_area("Her satıra bir haber metni", height=180)
        if st.button("TOPLU ANALİZ", disabled=not bulk_text.strip()):
            lines = [l.strip() for l in bulk_text.split("\n") if l.strip()]
            results = predict_batch(lines)
            for i, res in enumerate(results, 1):
                st.write(f"{i}. {res.get('label', 'ERROR')} - %{res.get('confidence', 0)*100:.2f}")

    with tab3:
        st.subheader("📋 Geçmiş")
        if user == "guest":
            st.info("Guest kullanıcılar için geçmiş tutulmaz.")
        else:
            rows = get_history(user)
            if not rows:
                st.info("Geçmiş kaydı yok.")
            for row in rows:
                with st.expander(f"[{row['timestamp']}] {row['prediction_label']} - %{row['confidence_score']:.1f}"):
                    st.write(row["text_input"])


if __name__ == "__main__":
    main()
