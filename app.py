
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from sqlalchemy import select
from database import init_db, SessionLocal, Person, FaceEmb, get_or_create_person, add_face, list_persons_with_counts, all_faces
from face_engine import FaceEngine
import datetime as dt

st.set_page_config(page_title="Face Recognition Web", layout="wide")

@st.cache_resource
def load_engine():
    return FaceEngine()

@st.cache_resource
def init_database():
    init_db()
    return True

engine_ready = init_database()
face_engine = load_engine()

st.title("🔎 Face Recognition (InsightFace + SQLite)")

tabs = st.tabs(["🔐 Recognize", "➕ Register Face", "🗂️ Database"])

# --------------- Recognize Tab ---------------
with tabs[0]:
    st.subheader("Deteksi & Identifikasi")
    src = st.radio("Sumber gambar:", ["Upload", "Kamera"], horizontal=True)
    img_bytes = None

    if src == "Upload":
        up = st.file_uploader("Upload foto (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if up:
            img_bytes = up.read()
    else:
        cam = st.camera_input("Ambil foto")
        if cam:
            img_bytes = cam.getvalue()

    if img_bytes:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        W, H = pil.size
        faces = face_engine.extract(pil)

        # Load DB embeddings
        session = SessionLocal()
        rows = session.query(FaceEmb, Person).join(Person, FaceEmb.person_id == Person.id).all()
        db_embs = []
        meta = []
        for fe, person in rows:
            emb = np.frombuffer(fe.embedding, dtype=np.float32)
            db_embs.append(emb)
            meta.append({"face_id": fe.id, "person_id": person.id, "name": person.name})
        if len(db_embs) > 0:
            db_embs = np.stack(db_embs, axis=0)
        else:
            db_embs = np.empty((0, 512), dtype=np.float32)

        draw = ImageDraw.Draw(pil)
        # Try to load a default font; Streamlit containers may not have many fonts
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()

        results = []
        for f in faces:
            idx, sim = face_engine.match(f.embedding, db_embs)
            if idx >= 0:
                name = meta[idx]["name"]
            else:
                name = "Unknown"

            label = f"{name} • {f.gender or '?'} • age: {f.age if f.age is not None and f.age>=0 else '?'} • sim: {sim:.2f}"
            x1, y1, x2, y2 = f.bbox
            draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=3)
            tw, th = draw.textlength(label, font=font), 18
            draw.rectangle([x1, max(0, y1-22), x1+tw+8, y1], fill=(0,255,0))
            draw.text((x1+4, y1-20), label, fill=(0,0,0), font=font)

            results.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "name": name,
                "gender": f.gender,
                "age": int(f.age) if f.age is not None and f.age>=0 else None,
                "similarity": float(f"{sim:.4f}")
            })

        c1, c2 = st.columns([2,1])
        with c1:
            st.image(pil, caption=f"Deteksi: {len(faces)} wajah", use_column_width=True)
        with c2:
            st.json({"faces": results})

# --------------- Register Tab ---------------
with tabs[1]:
    st.subheader("Tambah Wajah ke Database")
    with st.form("reg_form", clear_on_submit=False):
        name = st.text_input("Nama *", help="Nama identitas yang akan disimpan")
        info = st.text_input("Catatan (opsional)")
        src2 = st.radio("Sumber gambar:", ["Upload", "Kamera"], horizontal=True)
        imgs = []

        if src2 == "Upload":
            ups = st.file_uploader("Upload satu atau beberapa foto (JPG/PNG)", type=["jpg","jpeg","png"], accept_multiple_files=True)
            if ups:
                imgs = [u.read() for u in ups]
        else:
            cam2 = st.camera_input("Ambil foto")
            if cam2:
                imgs = [cam2.getvalue()]

        submitted = st.form_submit_button("Simpan")
        if submitted:
            if not name.strip():
                st.error("Nama wajib diisi.")
            elif not imgs:
                st.error("Minimal upload/ambil 1 foto.")
            else:
                session = SessionLocal()
                person = get_or_create_person(session, name=name.strip(), info=info.strip() if info else None)
                added = 0
                for b in imgs:
                    pil = Image.open(io.BytesIO(b)).convert("RGB")
                    faces = face_engine.extract(pil)
                    if len(faces) == 0:
                        st.warning("Tidak ada wajah terdeteksi pada salah satu gambar — dilewati.")
                        continue
                    # Ambil semua wajah yang ada (kalau lebih dari satu, semua disimpan)
                    for f in faces:
                        emb_bytes = f.embedding.astype(np.float32).tobytes()
                        face_img_bytes = f.crop_bytes
                        add_face(session, person_id=person.id, embedding_bytes=emb_bytes, face_img_bytes=face_img_bytes)
                        added += 1
                st.success(f"Berhasil menyimpan {added} face embedding untuk '{person.name}'.")

# --------------- Database Tab ---------------
with tabs[2]:
    st.subheader("Data Tersimpan")
    session = SessionLocal()
    persons = list_persons_with_counts(session)

    if len(persons) == 0:
        st.info("Belum ada data.")
    else:
        for person, cnt in persons:
            with st.expander(f"{person.name} — {cnt} wajah"):
                st.write(person.info or "-")
                # show latest faces
                faces = session.query(FaceEmb).filter(FaceEmb.person_id==person.id).order_by(FaceEmb.created_at.desc()).limit(12).all()
                cols = st.columns(6)
                i = 0
                for fe in faces:
                    with cols[i % 6]:
                        if fe.img:
                            st.image(fe.img, caption=f"face_id={fe.id}", use_column_width=True)
                        else:
                            st.caption(f"face_id={fe.id}")
                    i += 1

    st.divider()
    st.caption("Tip: untuk threshold akurasi, ubah konstanta SIM_THRESHOLD pada face_engine.py bila perlu.")
